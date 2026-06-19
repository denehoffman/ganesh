//! Internal duck-typing helpers for Python-facing config and run-option extraction.

use pyo3::{
    exceptions::PyAttributeError,
    prelude::FromPyObjectOwned,
    types::{PyAnyMethods, PyDict, PyDictMethods},
    Bound, PyAny, PyResult,
};

use crate::error::GaneshError;

pub(super) fn get_field<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        return dict.get_item(name);
    }

    match obj.getattr(name) {
        Ok(value) => Ok(Some(value)),
        Err(err) if err.is_instance_of::<PyAttributeError>(obj.py()) => Ok(None),
        Err(err) => Err(err),
    }
}

pub(super) fn extract_required_field<'py, T>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<T>
where
    T: FromPyObjectOwned<'py>,
    for<'a> <T as pyo3::FromPyObject<'a, 'py>>::Error: Into<pyo3::PyErr>,
{
    let Some(field) = get_field(obj, name)? else {
        return Err(
            GaneshError::ConfigError(format!("missing required Python field `{name}`")).into(),
        );
    };
    if field.is_none() {
        return Err(GaneshError::ConfigError(format!(
            "required Python field `{name}` must not be None"
        ))
        .into());
    }
    field.extract::<T>().map_err(Into::into)
}

pub(super) fn extract_optional_field<'py, T>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<T>>
where
    T: FromPyObjectOwned<'py>,
    for<'a> <T as pyo3::FromPyObject<'a, 'py>>::Error: Into<pyo3::PyErr>,
{
    match get_field(obj, name)? {
        Some(field) if !field.is_none() => field.extract::<T>().map(Some).map_err(Into::into),
        _ => Ok(None),
    }
}

pub(super) fn extract_optional_one_or_many_field<'py, T>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Vec<T>>>
where
    T: FromPyObjectOwned<'py>,
    for<'a> <T as pyo3::FromPyObject<'a, 'py>>::Error: Into<pyo3::PyErr>,
{
    match get_field(obj, name)? {
        Some(field) if !field.is_none() => field.extract::<Vec<T>>().map_or_else(
            |_| {
                field
                    .extract::<T>()
                    .map(|value| Some(vec![value]))
                    .map_err(Into::into)
            },
            |values| Ok(Some(values)),
        ),
        _ => Ok(None),
    }
}
