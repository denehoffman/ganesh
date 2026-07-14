use crate::{
    core::{
        mcmc_diagnostics::diagnostics_from_chain, EvalCounts, LinearAlgebra, MCMCDiagnostics,
        Matrix, NalgebraProvider, RealScalar, Vector,
    },
    traits::{ScalarBound, StatusMessage},
    DVector,
};
use serde::Serialize;
use serde_json::Error as SerdeJsonError;
use std::fmt::{Debug, Display};
use tabled::{
    builder::Builder,
    settings::{
        object::Row, style::HorizontalLine, themes::BorderCorrection, Alignment, Padding, Span,
        Style, Theme,
    },
};

pub(crate) fn unknown_uncertainties<T, B>(dimension: usize) -> Vector<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    Vector::from_vec(vec![T::literal(f64::NAN); dimension])
}

/// A trait used with the associated [`Summary`](`crate::traits::Algorithm`) type to set parameter names.
pub trait HasParameterNames: Sized {
    /// A mutable reference to the parameter names.
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>>;
    /// Set the names associated with each parameter.
    fn with_parameter_names<I, S>(mut self, parameter_names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        *self.get_parameter_names_mut() = Some(
            parameter_names
                .into_iter()
                .map(|s| s.as_ref().to_string())
                .collect(),
        );
        self
    }
}

/// A rendered summary containing both human-readable and machine-readable representations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedSummary {
    /// The summary rendered using its [`Display`] implementation.
    pub pretty: String,
    /// The summary rendered as JSON.
    pub json: String,
}

/// Helper methods for summary types that support both display formatting and serialization.
pub trait SummaryExport: Display + Serialize {
    /// Render the summary as a string using its [`Display`] implementation.
    fn to_pretty_string(&self) -> String {
        self.to_string()
    }

    /// Render the summary as a compact JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if the summary cannot be serialized to JSON.
    fn to_json_string(&self) -> Result<String, SerdeJsonError> {
        serde_json::to_string(self)
    }

    /// Render the summary as an indented JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if the summary cannot be serialized to JSON.
    fn to_json_string_pretty(&self) -> Result<String, SerdeJsonError> {
        serde_json::to_string_pretty(self)
    }

    /// Render the summary as both a display string and compact JSON in one call.
    ///
    /// # Errors
    ///
    /// Returns an error if the summary cannot be serialized to JSON.
    fn render(&self) -> Result<RenderedSummary, SerdeJsonError> {
        Ok(RenderedSummary {
            pretty: self.to_pretty_string(),
            json: self.to_json_string()?,
        })
    }

    /// Render the summary as both a display string and indented JSON in one call.
    ///
    /// # Errors
    ///
    /// Returns an error if the summary cannot be serialized to indented JSON.
    fn render_pretty_json(&self) -> Result<RenderedSummary, SerdeJsonError> {
        Ok(RenderedSummary {
            pretty: self.to_pretty_string(),
            json: self.to_json_string_pretty()?,
        })
    }
}
impl<T> SummaryExport for T where T: Display + Serialize {}

/// Scalar- and linear-algebra-generic result of a minimization run.
///
/// Serialization is deliberately not a core requirement: it is available to adapters when the
/// selected scalar and provider storage types support it.
#[derive(Clone, Serialize)]
#[serde(bound(
    serialize = "T: Serialize, B::VectorStorage: Serialize, B::MatrixStorage: Serialize"
))]
pub struct MinimizationSummary<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Optional user-facing parameter bounds.
    pub bounds: Option<Vec<ScalarBound<T>>>,
    /// Optional parameter names.
    pub parameter_names: Option<Vec<String>>,
    /// Final status message.
    pub message: StatusMessage,
    /// Initial parameters.
    pub x0: Vector<T, B>,
    /// Final parameters.
    pub x: Vector<T, B>,
    /// Parameter standard deviations.
    pub std: Vector<T, B>,
    /// Final objective value.
    pub fx: T,
    /// Evaluation counts requested by the algorithm.
    pub evals: EvalCounts,
    /// Covariance matrix.
    pub covariance: Matrix<T, B>,
}

impl<T, B> Default for MinimizationSummary<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            bounds: None,
            parameter_names: None,
            message: StatusMessage::default(),
            x0: Vector::zeros(0),
            x: Vector::zeros(0),
            std: Vector::zeros(0),
            fx: T::zero(),
            evals: EvalCounts::default(),
            covariance: Matrix::zeros(0, 0),
        }
    }
}

impl<T, B> HasParameterNames for MinimizationSummary<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

impl<T, B> Display for MinimizationSummary<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut builder = Builder::default();
        let status = if self.message.success() {
            "Converged".to_string()
        } else {
            "Not converged".to_string()
        };
        let parameter_row = |index: usize| {
            let name = self
                .parameter_names
                .as_ref()
                .and_then(|names| names.get(index))
                .cloned()
                .unwrap_or_else(|| format!("x_{index}"));
            let uncertainty = if index < self.std.len() {
                format!("{:.5}", self.std.get(index))
            } else {
                "—".to_string()
            };
            let initial = if index < self.x0.len() {
                format!("{:.5}", self.x0.get(index))
            } else {
                "—".to_string()
            };
            [
                name,
                format!("{:.5}", self.x.get(index)),
                uncertainty,
                initial,
            ]
        };
        let mut title = vec![String::new(); 6];
        title[0] = "MINIMIZATION SUMMARY".to_string();
        builder.push_record(title);
        builder.push_record(["Status", "f(x)", "# f(x)", "# ∇f(x)", "# H(x)", ""]);
        builder.push_record([
            status,
            format!("{:.5}", self.fx),
            self.evals.f().to_string(),
            self.evals.g().to_string(),
            self.evals.h().to_string(),
            String::new(),
        ]);
        builder.push_record(["Message", &self.message.to_string(), "", "", "", ""]);
        builder.push_record(["Parameters", "", "", "", "Bounds", ""]);
        builder.push_record(["Name", "Value", "Uncertainty", "Initial", "Lower", "Upper"]);
        for index in 0..self.x.len() {
            let mut row = parameter_row(index).to_vec();
            let bound = self
                .bounds
                .as_ref()
                .and_then(|bounds| bounds.get(index))
                .copied()
                .unwrap_or(ScalarBound::Unbounded);
            let (lower, upper) = match bound {
                ScalarBound::Unbounded => ("−∞".to_string(), "∞".to_string()),
                ScalarBound::Lower(lower) => (format!("{lower:.5}"), "∞".to_string()),
                ScalarBound::Upper(upper) => ("−∞".to_string(), format!("{upper:.5}")),
                ScalarBound::Both(lower, upper) => (format!("{lower:.5}"), format!("{upper:.5}")),
            };
            let endpoints: [String; 2] = (lower, upper).into();
            row.extend(endpoints);
            builder.push_record(row);
        }
        let mut table = builder.build();
        let mut theme = Theme::from_style(Style::rounded().remove_horizontals());
        for row in 1..=5 {
            theme.insert_horizontal_line(row, HorizontalLine::inherit(Style::modern()));
        }
        table
            .with(theme)
            .modify(
                Row::from(0),
                (Alignment::center(), Padding::new(1, 1, 0, 0)),
            )
            .modify((0, 0), Span::column(6))
            .modify((3, 1), Span::column(5))
            .modify((1, 4), Span::column(2))
            .modify((2, 4), Span::column(2))
            .modify((4, 0), Span::column(4))
            .modify((4, 4), Span::column(2))
            .with(BorderCorrection::span());
        f.write_str(&table.to_string())
    }
}

impl<T, B> Debug for MinimizationSummary<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Minimization Summary: status={}, f(x)={}, cost_evals={}, gradient_evals={}",
            self.message,
            self.fx,
            self.evals.f(),
            self.evals.g()
        )
    }
}

/// Scalar- and linear-algebra-generic MCMC result.
#[derive(Clone, Serialize)]
#[serde(bound(serialize = "B::VectorStorage: Serialize"))]
pub struct MCMCSummary<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Optional parameter names.
    pub parameter_names: Option<Vec<String>>,
    /// Final status message.
    pub message: StatusMessage,
    /// Retained positions grouped by walker.
    pub chain: Vec<Vec<Vector<T, B>>>,
    /// Evaluation counts requested by the sampler.
    pub evals: EvalCounts,
    /// `(walkers, retained steps, variables)`.
    pub dimension: (usize, usize, usize),
}

impl<T, B> Default for MCMCSummary<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            parameter_names: None,
            message: StatusMessage::default(),
            chain: Vec::new(),
            evals: EvalCounts::default(),
            dimension: (0, 0, 0),
        }
    }
}

impl<T, B> HasParameterNames for MCMCSummary<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

impl<T, B> Display for MCMCSummary<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut builder = Builder::default();
        builder.push_record(["MCMC SUMMARY", "", "", ""]);
        builder.push_record(["Status", "Walkers", "Retained steps", "Variables"]);
        builder.push_record([
            if self.message.success() {
                "Complete".to_string()
            } else {
                "Incomplete".to_string()
            },
            self.dimension.0.to_string(),
            self.dimension.1.to_string(),
            self.dimension.2.to_string(),
        ]);
        builder.push_record(["Message", &self.message.to_string(), "", ""]);
        builder.push_record([
            "Density evaluations",
            &self.evals.f().to_string(),
            "Retained samples",
            &(self.dimension.0.saturating_mul(self.dimension.1)).to_string(),
        ]);
        let mut table = builder.build();
        let mut theme = Theme::from_style(Style::rounded().remove_horizontals());
        for row in 1..=4 {
            theme.insert_horizontal_line(row, HorizontalLine::inherit(Style::modern()));
        }
        table
            .with(theme)
            .modify(
                Row::from(0),
                (Alignment::center(), Padding::new(1, 1, 0, 0)),
            )
            .modify((0, 0), Span::column(4))
            .modify((3, 1), Span::column(3))
            .with(BorderCorrection::span());
        f.write_str(&table.to_string())
    }
}

impl<T, B> Debug for MCMCSummary<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MCMC Summary: status={}, density_evals={}, dimension={:?}",
            self.message,
            self.evals.f(),
            self.dimension
        )
    }
}

impl<T, B> MCMCSummary<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Return a burned and thinned clone of the retained native chain.
    pub fn get_chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<Vec<Vector<T, B>>> {
        let burn = burn.unwrap_or(0);
        let thin = thin.unwrap_or(1).max(1);
        self.chain
            .iter()
            .map(|walker| walker.iter().skip(burn).step_by(thin).cloned().collect())
            .collect()
    }

    /// Return the burned and thinned chain flattened across walkers.
    pub fn get_flat_chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<Vector<T, B>> {
        self.get_chain(burn, thin).into_iter().flatten().collect()
    }

    /// Compute diagnostics through the crate's stable f64 reporting boundary.
    pub fn diagnostics(&self, burn: Option<usize>, thin: Option<usize>) -> MCMCDiagnostics {
        let chain = self
            .get_chain(burn, thin)
            .into_iter()
            .map(|walker| {
                walker
                    .into_iter()
                    .map(|position| {
                        DVector::from_vec(
                            position
                                .to_vec()
                                .into_iter()
                                .map(|value| value.to_f64().unwrap_or(f64::NAN))
                                .collect(),
                        )
                    })
                    .collect()
            })
            .collect::<Vec<Vec<DVector<f64>>>>();
        diagnostics_from_chain(&chain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_summaries_render_human_readable_output() {
        let mut result = MinimizationSummary::<f64> {
            x0: [2.0, -1.0].into(),
            x: [1.0, 0.5].into(),
            std: [0.1, 0.2].into(),
            fx: 0.25,
            evals: EvalCounts::new(12, 4, 1),
            ..Default::default()
        };
        result.message.succeed_with_message("minimum found");
        let display = result.to_string();
        assert!(display.contains("MINIMIZATION SUMMARY"));
        assert!(display.contains("Parameter"));
        assert!(display.contains("Uncertainty"));
        assert!(display.contains("x_0"));
        assert!(display.contains("minimum found"));
        assert!(display.contains("Bounds"));
        assert!(display.contains("# H(x)"));
        assert!(display.contains("−∞"));
        assert!(display.contains('∞'));
        assert!(!display.lines().next().unwrap().contains('┬'));
        let message_row = display
            .lines()
            .find(|line| line.contains("Message"))
            .unwrap();
        assert_eq!(message_row.matches('│').count(), 3);
        assert_eq!(
            format!("{result:?}"),
            "Minimization Summary: status=Success: minimum found, f(x)=0.25, cost_evals=12, gradient_evals=4"
        );
        let json = result.to_json_string().unwrap();
        assert!(json.contains("\"parameter_names\":null"));
        assert!(json.contains("\"bounds\":null"));
        assert!(json.contains("\"fx\":0.25"));

        let mut result = MCMCSummary::<f64> {
            evals: EvalCounts::new(240, 0, 0),
            dimension: (12, 20, 3),
            ..Default::default()
        };
        result.message.succeed_with_message("sampling complete");
        let display = result.to_string();
        assert!(display.contains("MCMC SUMMARY"));
        assert!(display.contains("Retained steps"));
        assert!(display.contains("240"));
        assert_eq!(
            format!("{result:?}"),
            "MCMC Summary: status=Success: sampling complete, density_evals=240, dimension=(12, 20, 3)"
        );
        let json = result.to_json_string().unwrap();
        assert!(json.contains("\"dimension\":[12,20,3]"));
    }

    #[test]
    fn empty_summaries_still_render_complete_tables() {
        let minimization = MinimizationSummary::<f64>::default().to_string();
        assert!(minimization.contains("MINIMIZATION SUMMARY"));
        assert!(minimization.contains("Parameter"));

        let mcmc = MCMCSummary::<f64>::default().to_string();
        assert!(mcmc.contains("MCMC SUMMARY"));
        assert!(mcmc.contains("Density evaluations"));
    }

    #[test]
    fn minimization_bounds_use_grouped_headers_and_unicode_infinity() {
        let summary = MinimizationSummary::<f64> {
            bounds: Some(vec![
                ScalarBound::Unbounded,
                ScalarBound::Lower(-2.0),
                ScalarBound::Upper(3.0),
                ScalarBound::Both(-4.0, 5.0),
            ]),
            x0: [0.0, 0.0, 0.0, 0.0].into(),
            x: [0.0, 0.0, 0.0, 0.0].into(),
            std: unknown_uncertainties(4),
            evals: EvalCounts::new(10, 2, 3),
            ..Default::default()
        };
        let display = summary.to_string();
        assert!(display.contains("Bounds"));
        assert!(display.contains("Lower"));
        assert!(display.contains("Upper"));
        assert!(display.contains("−∞"));
        assert!(display.contains('∞'));
        assert!(display.contains("# H(x)"));
        assert!(display.contains('3'));
        assert!(summary.bounds.as_ref().unwrap()[0] == ScalarBound::Unbounded);
        assert!(summary.to_json_string().unwrap().contains("\"bounds\""));
    }
}
