"""Launch helper that patches Gradio client utils bug before starting the app."""
import gradio_client.utils as gu

# Patch for Gradio 4.44.1 bug: additionalProperties can be bool, not dict
_orig_fn = gu._json_schema_to_python_type
def _patched_fn(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_fn(schema, defs)
gu._json_schema_to_python_type = _patched_fn

_orig_gt = gu.get_type
def _patched_gt(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig_gt(schema)
gu.get_type = _patched_gt

from socialscikit.ui.main_app import create_app
app = create_app()
app.launch(server_port=7860, server_name="0.0.0.0", share=False)
