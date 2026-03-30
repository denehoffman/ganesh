use crate::{
    DMatrix, DVector, Float,
    algorithms::mcmc::ChainStorageMode,
    core::transforms::Bounds,
    traits::{Bound, StatusMessage},
};
use serde::{Deserialize, Serialize};
use serde_json::Error as SerdeJsonError;
use std::fmt::Display;

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
    fn to_json_string(&self) -> Result<String, SerdeJsonError> {
        serde_json::to_string(self)
    }

    /// Render the summary as an indented JSON string.
    fn to_json_string_pretty(&self) -> Result<String, SerdeJsonError> {
        serde_json::to_string_pretty(self)
    }

    /// Render the summary as both a display string and compact JSON in one call.
    fn render(&self) -> Result<RenderedSummary, SerdeJsonError> {
        Ok(RenderedSummary {
            pretty: self.to_pretty_string(),
            json: self.to_json_string()?,
        })
    }

    /// Render the summary as both a display string and indented JSON in one call.
    fn render_pretty_json(&self) -> Result<RenderedSummary, SerdeJsonError> {
        Ok(RenderedSummary {
            pretty: self.to_pretty_string(),
            json: self.to_json_string_pretty()?,
        })
    }
}
impl<T> SummaryExport for T where T: Display + Serialize {}

/// A struct that holds the results of a minimization run.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MinimizationSummary {
    /// The bounds of the parameters. This is `None` if no bounds were set.
    pub bounds: Option<Bounds>,
    /// The names of the parameters. This is `None` if no names were set.
    pub parameter_names: Option<Vec<String>>,
    /// A message that can be set by minimization algorithms.
    pub message: StatusMessage,
    /// The initial parameters of the minimization.
    pub x0: DVector<Float>,
    /// The current parameters of the minimization.
    pub x: DVector<Float>,
    /// The standard deviations of the parameters at the end of the fit.
    pub std: DVector<Float>,
    /// The current value of the minimization problem function at [`MinimizationSummary::x`].
    pub fx: Float,
    /// The number of function evaluations.
    pub cost_evals: usize,
    /// The number of gradient evaluations.
    pub gradient_evals: usize,
    /// Covariance of fit parameters.
    pub covariance: DMatrix<Float>,
}

impl HasParameterNames for MinimizationSummary {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

impl Display for MinimizationSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use tabled::{
            builder::Builder,
            settings::{
                Alignment, Color, Padding, Span, Style, Theme, object::Row, style::HorizontalLine,
                themes::BorderCorrection,
            },
        };
        let mut builder = Builder::default();
        builder.push_record(["FIT RESULTS"]);
        builder.push_record(["Status", "f(x)", "", "#f(x)", "", "#∇f(x)", ""]);
        builder.push_record([
            if self.message.success() {
                "Converged"
            } else {
                "Invalid Minimum"
            },
            &format!("{:.5}", self.fx),
            "",
            &format!("{:.5}", self.cost_evals),
            "",
            &format!("{:.5}", self.gradient_evals),
            "",
        ]);
        builder.push_record(["Message", &self.message.to_string()]);

        let names = self
            .parameter_names
            .clone()
            .unwrap_or_else(|| {
                vec![""; self.x.len()]
                    .into_iter()
                    .enumerate()
                    .map(|(i, _)| format!("x_{}", i))
                    .collect::<Vec<_>>()
            })
            .into_iter();
        let bounds = self
            .bounds
            .clone()
            .map(|bs| bs.iter().map(|b| b.0).collect())
            .unwrap_or_else(|| vec![Bound::NoBound; self.x.len()])
            .into_iter();

        builder.push_record(["Parameter", "", "", "", "Bound", "", "At Limit?"]);

        builder.push_record(["", "=", "σ", "0", "-", "+", ""]);
        for ((((v, v0), e), b), n) in self
            .x
            .iter()
            .zip(&self.x0)
            .zip(&self.std)
            .zip(bounds)
            .zip(names)
        {
            builder.push_record([
                &n,
                &format!("{:.5}", v),
                &format!("{:.5}", e),
                &format!("{:.5}", v0),
                &format!("{:.5}", b.lower()),
                &format!("{:.5}", b.upper()),
                &(if b.at_bound(*v, Float::EPSILON) {
                    "Yes"
                } else {
                    "No"
                }
                .to_string()),
            ]);
        }
        let mut table = builder.build();
        let mut style = Theme::from_style(Style::rounded().remove_horizontals());
        style.insert_horizontal_line(1, HorizontalLine::inherit(Style::modern()));
        style.insert_horizontal_line(2, HorizontalLine::inherit(Style::modern()));
        style.insert_horizontal_line(3, HorizontalLine::inherit(Style::modern()));
        style.insert_horizontal_line(4, HorizontalLine::inherit(Style::modern()));
        style.insert_horizontal_line(5, HorizontalLine::inherit(Style::modern()));
        style.insert_horizontal_line(6, HorizontalLine::inherit(Style::modern()));

        table
            .with(style)
            .modify(
                Row::from(0),
                (Padding::new(1, 1, 1, 1), Alignment::center(), Color::BOLD),
            )
            .modify((0, 0), Span::column(7))
            .modify(Row::from(1), Color::BOLD)
            .modify((1, 1), Span::column(2))
            .modify((1, 3), Span::column(2))
            .modify((1, 5), Span::column(2))
            .modify((2, 1), Span::column(2))
            .modify((2, 3), Span::column(2))
            .modify((2, 5), Span::column(2))
            .modify(Row::from(3), Padding::new(1, 1, 1, 1))
            .modify((3, 0), Color::BOLD)
            .modify((3, 1), Span::column(6))
            .modify(Row::from(4), Color::BOLD)
            .modify((4, 0), Span::column(4))
            .modify((4, 4), Span::column(2))
            .modify(Row::from(5), Color::BOLD)
            .with(BorderCorrection::span());

        f.write_str(&table.to_string())?;
        Ok(())
    }
}

/// A struct that holds the results of a minimization run.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulatedAnnealingSummary<I> {
    /// The bounds of the parameters. This is `None` if no bounds were set.
    pub bounds: Option<Bounds>,
    /// A message that can be set by minimization algorithms.
    pub message: StatusMessage,
    /// The initial parameters of the minimization.
    pub x0: I,
    /// The current parameters of the minimization.
    pub x: I,
    /// The standard deviations of the parameters at the end of the fit.
    pub fx: Float,
    /// The number of function evaluations.
    pub cost_evals: usize,
}

/// A struct that holds the results of an MCMC sampling.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MCMCSummary {
    /// The bounds of the parameters. This is `None` if no bounds were set.
    pub bounds: Option<Bounds>,
    /// The names of the parameters. This is `None` if no names were set.
    pub parameter_names: Option<Vec<String>>,
    /// A message that can be set by minimization algorithms.
    pub message: StatusMessage,
    /// The chain of positions sampled by each walker with dimension `(n_walkers, n_steps,
    /// n_variables)`.
    pub chain: Vec<Vec<DVector<Float>>>,
    /// The mode used to retain chain history in memory during sampling.
    pub chain_storage: ChainStorageMode,
    /// The number of function evaluations.
    pub cost_evals: usize,
    /// The number of gradient evaluations.
    pub gradient_evals: usize,
    /// The dimension of the ensemble `(n_walkers, n_steps, n_variables)`
    pub dimension: (usize, usize, usize),
}

impl MCMCSummary {
    /// Get a [`Vec`] containing a [`Vec`] of positions for each
    /// [`Walker`](crate::algorithms::mcmc::Walker) in the ensemble
    ///
    /// If `burn` is [`None`], no burn-in will be performed, otherwise the given number of steps
    /// will be discarded from the beginning of each [`Walker`](crate::algorithms::mcmc::Walker)'s history.
    ///
    /// If `thin` is [`None`], no thinning will be performed, otherwise every `thin`-th step will
    /// be discarded from the [`Walker`](crate::algorithms::mcmc::Walker)'s history.
    pub fn get_chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<Vec<DVector<Float>>> {
        let burn = burn.unwrap_or(0);
        let thin = thin.unwrap_or(1);
        self.chain
            .iter()
            .map(|walker| {
                walker
                    .iter()
                    .skip(burn)
                    .enumerate()
                    .filter_map(|(i, position)| {
                        if i % thin == 0 {
                            Some(position.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect()
    }
    /// Get a [`Vec`] containing positions for each [`Walker`](crate::algorithms::mcmc::Walker) in the ensemble, flattened
    ///
    /// If `burn` is [`None`], no burn-in will be performed, otherwise the given number of steps
    /// will be discarded from the beginning of each [`Walker`](crate::algorithms::mcmc::Walker)'s history.
    ///
    /// If `thin` is [`None`], no thinning will be performed, otherwise every `thin`-th step will
    /// be discarded from the [`Walker`](crate::algorithms::mcmc::Walker)'s history.
    pub fn get_flat_chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<DVector<Float>> {
        let chain = self.get_chain(burn, thin);
        chain.into_iter().flatten().collect()
    }
}

impl HasParameterNames for MCMCSummary {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

impl Display for MCMCSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MCMC Summary: status={}, cost_evals={}, gradient_evals={}, dimension={:?}",
            self.message, self.cost_evals, self.gradient_evals, self.dimension
        )
    }
}

impl<I> Display for SimulatedAnnealingSummary<I>
where
    I: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Simulated Annealing Summary: status={}, f(x)={:.5}, cost_evals={}",
            self.message, self.fx, self.cost_evals
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_minimization_result() {
        let result = MinimizationSummary {
            bounds: None,
            parameter_names: None,
            message: StatusMessage::default().set_success(),
            x0: dvector![1.0, 2.0, 3.0],
            x: dvector![1.0, 2.0, 3.0],
            std: dvector![0.1, 0.2, 0.3],
            fx: 3.0,
            cost_evals: 10,
            gradient_evals: 5,
            covariance: dmatrix![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        println!("{}", result);
    }

    #[test]
    fn minimization_summary_can_render_pretty_and_json() {
        let result = MinimizationSummary {
            bounds: None,
            parameter_names: Some(vec!["alpha".to_string(), "beta".to_string()]),
            message: StatusMessage::default().set_success_with_message("ok"),
            x0: dvector![1.0, 2.0],
            x: dvector![0.5, 1.5],
            std: dvector![0.1, 0.2],
            fx: 1.25,
            cost_evals: 10,
            gradient_evals: 4,
            covariance: dmatrix![1.0, 0.0, 0.0, 1.0],
        };

        let rendered = result.render().unwrap();

        assert!(rendered.pretty.contains("FIT RESULTS"));
        assert!(rendered.json.contains("\"fx\":1.25"));
        assert!(rendered.json.contains("\"parameter_names\":[\"alpha\",\"beta\"]"));
    }

    #[test]
    fn mcmc_summary_can_render_pretty_json() {
        let result = MCMCSummary {
            bounds: None,
            parameter_names: Some(vec!["x".to_string()]),
            message: StatusMessage::default().set_initialized_with_message("warmup"),
            chain: vec![vec![dvector![1.0], dvector![2.0]]],
            chain_storage: ChainStorageMode::Full,
            cost_evals: 8,
            gradient_evals: 0,
            dimension: (1, 2, 1),
        };

        let rendered = result.render_pretty_json().unwrap();

        assert!(rendered.pretty.contains("MCMC Summary"));
        assert!(rendered.pretty.contains("cost_evals=8"));
        assert!(rendered.json.contains("\n  \"dimension\": [\n"));
        assert!(rendered.json.contains("\"cost_evals\": 8"));
    }

    #[test]
    fn simulated_annealing_summary_can_render_json() {
        let result = SimulatedAnnealingSummary {
            bounds: None,
            message: StatusMessage::default().set_success_with_message("done"),
            x0: "start".to_string(),
            x: "finish".to_string(),
            fx: 0.5,
            cost_evals: 12,
        };

        let rendered = result.render().unwrap();

        assert!(rendered.pretty.contains("Simulated Annealing Summary"));
        assert!(rendered.json.contains("\"fx\":0.5"));
        assert!(rendered.json.contains("\"x\":\"finish\""));
    }
}
