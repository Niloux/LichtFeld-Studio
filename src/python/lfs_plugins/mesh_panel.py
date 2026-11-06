# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesh extraction controls; the work runs on the native collision job."""
from pathlib import Path

import lichtfeld as lf
from . import rml_widgets as w
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .ui import invalidate_panel, RuntimeState, PanelStateBinding
from .panels import panel_class
from .types import Panel

__lfs_panel_classes__ = ["MeshPanel"]
__lfs_panel_ids__ = ["lfs.mesh"]

SPECS = {
    "voxel_size": ScrubFieldSpec(.0001, 10, .001, "%.4g"),
    "view_stride": ScrubFieldSpec(1, 64, 1, "%.0f"),
    "full_weight_range": ScrubFieldSpec(0, 1000, .1, "%.3g"),
    "min_views": ScrubFieldSpec(1, 64, 1, "%.0f"),
    "simplify_error": ScrubFieldSpec(0, .05, .001, "%.3g"),
    "max_range": ScrubFieldSpec(0, 1000, 1, "%.3g"),
    "texel_size": ScrubFieldSpec(0, .05, .001, "%.3g"),
}
LABELS = {"voxel_size": "main_panel.voxel_size", "view_stride": "mesh.view_stride",
          "full_weight_range": "mesh.full_weight_range", "min_views": "mesh.min_views",
          "simplify_error": "mesh.simplify_error", "max_range": "mesh.max_range", "texel_size": "mesh.texel_size"}
# Which container each field goes in. The voxel size also sets how far a photo may see past a surface when baking,
# so it stays when the surface comes from a file.
CONTAINERS = {"voxel_size": "mesh-shared-fields", "texel_size": "mesh-texture-fields"}


@panel_class("mesh")
class MeshPanel(Panel):
    def __init__(self):
        self.handle = None
        self.values = dict(voxel_size=.02, view_stride=2, full_weight_range=1., min_views=3, simplify_error=.005, max_range=0., texel_size=0.,
                           use_file=False, file="", error="")
        self.state = {}
        self.close_when_done = False
        self.scrubs = ScrubFieldController(
            SPECS, self.values.__getitem__, self.set_value,
            on_error=lambda error: self.set_value("error", lf.ui.tr("preferences.viewport_defaults_invalid")),
        )

    def set_value(self, prop, value):
        self.values[prop] = value
        invalidate_panel(self.handle, "*")

    def textured(self):
        return self.values["texel_size"] > 0

    def notice(self):
        """Why the settings cannot run as they are, or empty. A missing file shows on its own row."""
        if self.values["use_file"] and not self.textured():
            return lf.ui.tr("mesh.needs_texel")
        return ""

    def output(self):
        return lf.ui.tr("mesh.output_osgb" if self.textured() else "mesh.output_ply")

    def ready(self):
        if self.values["use_file"] and not self.values["file"]:
            return False
        return bool(self.state.get("source_name")) and not self.state.get("active", False) and not self.notice()

    def choose_file(self, *_):
        start = str(Path(self.values["file"]).parent) if self.values["file"] else ""
        path = lf.ui.open_mesh_file_dialog(start)
        if not path:
            return
        if Path(path).suffix.lower() != ".ply":
            self.values["error"] = lf.ui.tr("mesh.ply_only")
        else:
            self.values["error"] = ""
            self.values["file"] = path
        invalidate_panel(self.handle, "*")

    def generate(self, *_):
        if not self.ready():
            return
        name = Path(self.values["file"]).stem + "_baked" if self.values["use_file"] else self.state.get("source_name", "") + "_mesh"
        path = lf.ui.save_ply_file_dialog(name)
        if not path:
            return
        self.values["error"] = ""
        try:
            lf.generate_mesh(path, self.values["voxel_size"], int(self.values["view_stride"]),
                             self.values["full_weight_range"], int(self.values["min_views"]),
                             self.values["simplify_error"], self.values["max_range"], self.values["texel_size"],
                             self.values["file"] if self.values["use_file"] else "")
            self.close_when_done = True
        except (RuntimeError, ValueError) as exc:
            key = getattr(exc, "message_key", "")
            details = getattr(exc, "message_details", "")
            self.values["error"] = (lf.ui.tr(key) + (f"\n{details}" if details else "")) if key else lf.ui.tr("selection.operation_failed_generic")
        invalidate_panel(self.handle, "*")

    def cancel(self, *_):
        self.close_when_done = False
        lf.cancel_collision()

    def update(self, doc=None):
        state = lf.get_collision_state()
        changed = state != self.state
        self.state = state
        if self.close_when_done and not state.get("active", False):
            self.close_when_done = False
            if not state.get("error"):
                lf.ui.set_panel_enabled("lfs.mesh", False)
        return self.scrubs.sync_all() or changed

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("mesh")
        self.handle = model.get_handle()
        self.update()
        model.bind_func("panel_label", lambda: lf.ui.tr("mesh.title"))
        model.bind_func("mesh_source", lambda: self.state.get("source_name", ""))
        model.bind_func("mesh_active", lambda: self.state.get("active", False))
        model.bind_func("mesh_progress", lambda: self.state.get("progress", 0))
        model.bind_func("mesh_error", lambda: self.values["error"] or self.state.get("error", ""))
        model.bind_func("mesh_status", lambda: f'{self.state.get("stage") or lf.ui.tr("training_panel.running")} {self.state.get("progress", 0):.0%}')
        model.bind_func("mesh_file", lambda: self.values["file"])
        model.bind_func("mesh_file_name", lambda: Path(self.values["file"]).name if self.values["file"] else lf.ui.tr("mesh.no_file"))
        model.bind_func("mesh_output", self.output)
        model.bind_func("mesh_notice", self.notice)
        model.bind_func("mesh_ready", self.ready)
        model.bind("mesh_use_file", lambda: self.values["use_file"], lambda value: self.set_value("use_file", bool(value)))
        model.bind_event("mesh_choose_file", self.choose_file)
        model.bind_event("mesh_generate", self.generate)
        model.bind_event("mesh_cancel", self.cancel)
        model.bind_event("mesh_close", lambda *_: lf.ui.set_panel_enabled("lfs.mesh", False))

    def on_mount(self, doc):
        super().on_mount(doc)
        self.update()
        for key, spec in SPECS.items():
            container = doc.get_element_by_id(CONTAINERS.get(key, "mesh-fields"))
            if container is None:
                continue
            row = w.slider(container, key, label=lf.ui.tr(LABELS[key]), min=spec.min_value, max=spec.max_value,
                           step=spec.step, value=self.values[key])
            row.query_selector(".setting-row__label-col").set_attribute("data-lfs-i18n", LABELS[key])
            row.query_selector("input").set_attribute("data-value", key)
            row.set_attribute("data-tooltip", "tooltip.mesh." + key)
        self.scrubs.mount(doc)
        self.binding = PanelStateBinding(self.handle).watch(
            RuntimeState.scene_generation, RuntimeState.selection_generation,
            RuntimeState.viewport_toolbar_generation,
            refresh=self.update, dirty="*").watch(RuntimeState.language_generation, dirty="*")

    on_update = update

    def on_unmount(self, doc):
        self.close_when_done = False
        self.binding.close()
        self.scrubs.unmount()
        self.handle = None
        doc.remove_data_model("mesh")
