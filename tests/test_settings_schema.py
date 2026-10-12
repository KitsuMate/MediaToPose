"""The post-processing settings schema matches the stored RNA settings."""

from __future__ import annotations

import importlib
import math
import unittest

from tests.blender_test_support import register_addon


class SettingsSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        cls.schema = importlib.import_module(
            f"{cls.addon.__name__}.postprocess.settings_schema"
        )
        cls.rna = cls.addon.properties.MTPPostProcessingModuleConfig.bl_rna.properties

    def test_every_parameter_matches_its_rna_property(self) -> None:
        kinds = {
            "BOOLEAN": "bool",
            "FLOAT": "float",
            "INT": "int",
            "ENUM": "enum",
            "STRING": "string",
        }
        for name, parameter in self.schema.PARAMETERS.items():
            with self.subTest(parameter=name):
                prop = self.rna[name]
                self.assertEqual(kinds[prop.type], parameter.kind)
                self.assertEqual(prop.name, parameter.label)
                self.assertEqual(prop.description, parameter.description)
                if parameter.kind == "float":
                    self.assertTrue(
                        math.isclose(prop.default, parameter.default, rel_tol=1e-6)
                    )
                    self.assertTrue(
                        math.isclose(prop.hard_min, parameter.minimum, abs_tol=1e-6)
                    )
                    self.assertTrue(
                        math.isclose(prop.hard_max, parameter.maximum, abs_tol=1e-6)
                    )
                    self.assertEqual(prop.subtype, parameter.subtype or "NONE")
                elif parameter.kind == "enum":
                    self.assertEqual(prop.default, parameter.default)
                    self.assertEqual(
                        [
                            (item.identifier, item.name, item.description)
                            for item in prop.enum_items
                        ],
                        list(parameter.items),
                    )
                else:
                    self.assertEqual(prop.default, parameter.default)

    def test_modules_list_every_parameter_once_per_module(self) -> None:
        used = set()
        for module in self.schema.MODULES:
            self.assertEqual(len(module.parameters), len(set(module.parameters)))
            used.update(module.parameters)
        self.assertEqual(used, set(self.schema.PARAMETERS))

    def test_validation_coerces_and_rejects(self) -> None:
        errors = self.addon.core.errors
        values = self.schema.validate_settings(
            "TEMPORAL_CORRECTION", {"smoothing_strength": 1, "smooth_motion": True}
        )
        self.assertEqual(values, {"smoothing_strength": 1.0, "smooth_motion": True})
        with self.assertRaises(errors.InvalidRequest):
            self.schema.validate_settings("TEMPORAL_CORRECTION", {"strength": 1.0})
        with self.assertRaises(errors.InvalidRequest):
            self.schema.validate_settings(
                "TEMPORAL_CORRECTION", {"smoothing_strength": 2.0}
            )
        with self.assertRaises(errors.InvalidRequest):
            self.schema.validate_settings(
                "DETECTION_FILTER", {"manual_bad_scope": "TAIL"}
            )


if __name__ == "__main__":
    unittest.main()
