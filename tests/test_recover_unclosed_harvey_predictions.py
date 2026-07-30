import unittest

from evaluation.recover_unclosed_harvey_predictions import (
    compute_harvey_metrics,
    dedupe_recovered_entities,
    extract_complete_location_values,
    recover_prediction,
)


class RecoveryTests(unittest.TestCase):
    def test_recovers_complete_evidence_values_and_drops_truncated_tail(self):
        raw = (
            '[{"evidence":["Houston"],'
            '{"evidence":["I-45 & N. Main Street"],'
            '{"evidence":["unfinished"'
        )
        entities, stats = extract_complete_location_values(raw)
        self.assertEqual(
            entities,
            [
                {"evidence": ["Houston"]},
                {"evidence": ["I-45 & N. Main Street"]},
            ],
        )
        self.assertEqual(stats["complete_values"], 2)
        self.assertEqual(stats["incomplete_values"], 1)

    def test_scanner_handles_escaped_quotes(self):
        raw = '[{"evidence":["St. \\"Joseph\\" Hall"],{"evidence":["Texas"]'
        entities, _ = extract_complete_location_values(raw)
        self.assertEqual(
            entities,
            [
                {"evidence": ['St. "Joseph" Hall']},
                {"evidence": ["Texas"]},
            ],
        )

    def test_scanner_is_confined_to_first_array_region(self):
        raw = (
            '"evidence":["prefix"] '
            '[{"evidence":["Houston"]}] '
            '{"evidence":["suffix"]}'
        )
        entities, _ = extract_complete_location_values(raw)
        self.assertEqual(entities, [{"evidence": ["Houston"]}])

    def test_scanner_rejects_nested_metadata_field(self):
        raw = (
            '[{"meta":{"other":1,"evidence":["nested"]},'
            '"evidence":["Houston"]'
        )
        entities, _ = extract_complete_location_values(raw)
        self.assertEqual(entities, [{"evidence": ["Houston"]}])

    def test_scanner_accepts_gemma_repeated_key_form(self):
        raw = '[{"evidence":["Houston"],"evidence":["Texas"]'
        entities, _ = extract_complete_location_values(raw)
        self.assertEqual(
            entities,
            [
                {"evidence": ["Houston"]},
                {"evidence": ["Texas"]},
            ],
        )

    def test_scanner_requires_an_array(self):
        entities, stats = extract_complete_location_values(
            'No array. "evidence": "prose field"'
        )
        self.assertEqual(entities, [])
        self.assertEqual(stats["recognized_keys"], 0)

    def test_scanner_rejects_value_without_json_member_boundary(self):
        raw = '[{"evidence":["not complete"]junk'
        entities, stats = extract_complete_location_values(raw)
        self.assertEqual(entities, [])
        self.assertEqual(stats["incomplete_values"], 1)

    def test_scanner_recovers_qwen_missing_evidence_list_closer(self):
        raw = '[{"evidence":["Houston"}]'
        entities, stats = extract_complete_location_values(raw)
        self.assertEqual(entities, [{"evidence": ["Houston"]}])
        self.assertEqual(stats["missing_list_closer_values"], 1)
        self.assertEqual(stats["incomplete_values"], 0)

    def test_qwen_fallback_accepts_only_complete_scalar_items(self):
        raw = '[{"evidence":["Texas","Houston"}]'
        entities, stats = extract_complete_location_values(raw)
        self.assertEqual(entities, [{"evidence": ["Texas", "Houston"]}])
        self.assertEqual(stats["missing_list_closer_values"], 1)

    def test_qwen_fallback_rejects_missing_object_boundary(self):
        raw = '[{"evidence":["Houston"junk'
        entities, stats = extract_complete_location_values(raw)
        self.assertEqual(entities, [])
        self.assertEqual(stats["incomplete_values"], 1)

    def test_valid_postprocessed_array_is_preserved(self):
        prediction, audit = recover_prediction(
            pred_raw='[{"evidence":["ignored suffix"]',
            pred_postproc_raw='[{"evidence":["Austin"]}]',
            dedupe_policy="exact",
        )
        self.assertEqual(prediction, [{"evidence": ["Austin"]}])
        self.assertEqual(audit["mode"], "existing_valid_array")

    def test_exact_deduplication_is_case_sensitive(self):
        entities = [
            {"evidence": ["Texas"]},
            {"evidence": ["Texas"]},
            {"evidence": ["texas"]},
        ]
        recovered, dropped = dedupe_recovered_entities(entities, "exact")
        self.assertEqual(len(recovered), 2)
        self.assertEqual(dropped, 1)

    def test_normalized_deduplication_ignores_case_and_whitespace(self):
        entities = [
            {"evidence": ["Texas"]},
            {"evidence": [" texas  "]},
        ]
        recovered, dropped = dedupe_recovered_entities(entities, "normalized")
        self.assertEqual(recovered, [{"evidence": ["Texas"]}])
        self.assertEqual(dropped, 1)

    def test_metric_logic_matches_gold_category_stratification(self):
        predictions = [[{"evidence": ["Houston"]}, {"evidence": ["noise"]}]]
        golds = [
            [
                {
                    "evidence": ["Houston"],
                    "locationCate": "C9",
                }
            ]
        ]
        metrics, category_rows, counts = compute_harvey_metrics(
            predictions,
            golds,
            evidence_match_threshold=0.75,
        )
        self.assertEqual(counts, {"tp": 1.0, "fp": 1.0, "fn": 0.0})
        self.assertAlmostEqual(metrics["locdesc_precision"], 0.5)
        self.assertAlmostEqual(metrics["locdesc_recall"], 1.0)
        self.assertAlmostEqual(metrics["locdesc_f1"], 2.0 / 3.0)
        self.assertEqual(category_rows[0]["category"], "C9")
        self.assertEqual(category_rows[0]["fp"], 0)
        self.assertAlmostEqual(category_rows[0]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
