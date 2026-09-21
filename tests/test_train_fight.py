import argparse
import copy
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from campus_safety_ai.apps import train_fight
from campus_safety_ai.training.manifest import FightSample, write_manifest
from campus_safety_ai.training.metrics import Metrics


def arguments(root, **updates):
    values = dict(project_root=root, manifest=Path("manifest.csv"), output_dir=Path("experiment"),
                  epochs=5, batch_size=2, learning_rate=1e-4, weight_decay=1e-4, frames=8,
                  image_size=224, workers=0, device="cpu", seed=7, no_pretrained=True,
                  max_train_samples=None, max_val_samples=None)
    values.update(updates)
    return argparse.Namespace(**values)


def metric(f1=0.8, loss=0.5):
    return Metrics(loss=loss, accuracy=f1, balanced_accuracy=f1, precision=f1, recall=f1, f1=f1)


def checkpoint():
    return {"model_state": {"weight": 1}, "model_name": "fight-tsn-mobilenet-v3-small",
            "frame_count": 8, "image_size": 224, "class_names": ["non_fight", "fight"]}


class TrainFightTests(unittest.TestCase):
    def manifest(self, root):
        samples = []
        for split in ("train", "val", "test"):
            for label in (0, 1):
                path = f"{split}-{label}.mp4"
                (root / path).touch()
                samples.append(FightSample(path, label, "scfd", f"{split}-{label}", split))
        write_manifest(samples, root / "manifest.csv")
        return samples

    def mock_training(self, stack, epoch_metrics):
        torch = types.ModuleType("torch")
        torch.__version__ = "test-version"
        torch.manual_seed = MagicMock()
        torch.float32 = "float32"
        torch.tensor = MagicMock(side_effect=lambda value, **_: value)
        torch.load = MagicMock(side_effect=lambda path, **_: json.loads(Path(path).read_text()))
        torch.save = MagicMock(side_effect=lambda value, path: Path(path).write_text(json.dumps(value)))
        torch.nn = types.SimpleNamespace(CrossEntropyLoss=MagicMock(return_value=object()))
        optimizer = types.SimpleNamespace(param_groups=[{"lr": 1e-4}])
        scheduler = MagicMock()
        torch.optim = types.SimpleNamespace(
            AdamW=MagicMock(return_value=optimizer),
            lr_scheduler=types.SimpleNamespace(ReduceLROnPlateau=MagicMock(return_value=scheduler)),
        )
        data = types.ModuleType("torch.utils.data")
        data.DataLoader = MagicMock(side_effect=lambda dataset, **_: dataset)
        stack.enter_context(patch.dict(sys.modules, {"torch": torch, "torch.utils.data": data}))
        model = MagicMock()
        model.to.return_value = model
        model.state_dict.return_value = {"weight": 1}
        factory = stack.enter_context(patch.object(train_fight, "FightTsn", return_value=model))
        stack.enter_context(patch.object(train_fight, "_select_device", return_value="cpu"))
        dataset = stack.enter_context(patch.object(
            train_fight, "FightVideoDataset", side_effect=lambda root, samples, *args: samples,
        ))
        epochs = stack.enter_context(patch.object(train_fight, "_run_epoch", side_effect=epoch_metrics))
        return torch, factory, dataset, epochs

    def test_limit_is_reproducible_and_represents_all_dataset_labels(self):
        samples = [FightSample(f"{dataset}/{label}/{index}.mp4", label, dataset,
                               f"{dataset}:{label}:{index}", "train")
                   for dataset in ("scfd", "airtlab") for label in (0, 1) for index in range(30)]
        selected = train_fight._limit(samples, 20, seed=31)
        self.assertEqual(selected, train_fight._limit(list(reversed(samples)), 20, seed=31))
        self.assertNotEqual(selected, train_fight._limit(samples, 20, seed=32))
        self.assertEqual(len(selected), 20)
        self.assertEqual({(sample.dataset, sample.label) for sample in selected},
                         {("scfd", 0), ("scfd", 1), ("airtlab", 0), ("airtlab", 1)})
        self.assertEqual(len({sample.video_path for sample in selected}), 20)
        with self.assertRaisesRegex(ValueError, "every dataset/label"):
            train_fight._limit(samples, 3)

    def test_manifest_rejects_path_alias_and_group_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = self.manifest(root)
            leaked_path = replace(samples[2], video_path="./train-0.mp4")
            with self.assertRaisesRegex(ValueError, "path leaks"):
                train_fight._validate_manifest([samples[0], leaked_path], root)
            leaked_group = replace(samples[2], group_id=samples[0].group_id)
            with self.assertRaisesRegex(ValueError, "group_id leaks"):
                train_fight._validate_manifest([samples[0], leaked_group], root)

    def test_occupied_output_is_rejected_before_torch_or_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "experiment"
            output.mkdir()
            existing = output / "best.pt"
            existing.write_bytes(b"preserved model")
            with patch.dict(sys.modules, {"torch": None}), patch.object(train_fight, "FightTsn") as model:
                with self.assertRaisesRegex(ValueError, "existing experiment is preserved"):
                    train_fight.train(arguments(root))
                model.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"preserved model")
            self.assertEqual(list(output.iterdir()), [existing])

    def test_invalid_parameters_fail_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for update in ({"epochs": 0}, {"batch_size": -1}, {"patience": -1},
                           {"learning_rate": float("nan")}, {"weight_decay": -1},
                           {"max_train_samples": 0}, {"frames": 0}, {"workers": -1},
                           {"accumulation_steps": 0}, {"accumulation_steps": 1.5},
                           {"freeze_batch_norm": "yes"}):
                with self.subTest(update=update), self.assertRaises(ValueError):
                    train_fight.train(arguments(root, **update))
            self.assertEqual(list(root.iterdir()), [])

    def test_checkpoint_shape_and_class_mismatch_are_rejected(self):
        for update in ({"frame_count": 4}, {"image_size": 128}, {"class_names": ["fight", "non_fight"]}):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "does not match"):
                train_fight._validate_checkpoint(checkpoint() | update, 8, 224)

    def test_fine_tuning_keeps_baseline_and_stops_using_validation(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.manifest(root)
            init = root / "initial.pt"
            init.write_text(json.dumps(checkpoint()))
            # Training improves, but validation never beats the initialization.
            torch, factory, datasets, epochs = self.mock_training(
                stack, [metric(.9, .3), metric(.98), metric(.7), metric(.99), metric(.8)],
            )
            best = train_fight.train(arguments(root, init_checkpoint=init, patience=2))
            run = json.loads((best.parent / "run.json").read_text())
            saved = json.loads(best.read_text())
            self.assertEqual(run["status"], "early_stopped")
            self.assertEqual(run["epochs_completed"], 2)
            self.assertEqual(run["best_epoch"], 0)
            self.assertEqual(saved["best_epoch"], 0)
            self.assertEqual(saved["val_metrics"]["f1"], .9)
            self.assertEqual(run["initialization"]["checkpoint_sha256"], train_fight._sha256(init))
            self.assertEqual(run["manifest_sha256"], train_fight._sha256(root / "manifest.csv"))
            self.assertEqual(saved["provenance"]["run_id"], run["run_id"])
            self.assertEqual(run["torch_version"], "test-version")
            self.assertEqual(factory.call_args.kwargs["pretrained"], False)
            self.assertEqual(epochs.call_count, 5)
            self.assertEqual({s.split for call in datasets.call_args_list for s in call.args[1]}, {"train", "val"})
            self.assertEqual(torch.optim.lr_scheduler.ReduceLROnPlateau.call_args.kwargs["patience"], 1)
            records = [json.loads(line) for line in (best.parent / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([record["improved"] for record in records], [False, False])
            self.assertTrue(all(record["epoch_seconds"] >= 0 for record in records))
            self.assertTrue(all(record["effective_batch_size"] == 2 for record in records))

    def test_equal_f1_uses_lower_validation_loss_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.manifest(root)
            self.mock_training(stack, [metric(.99), metric(.8, .6), metric(.95), metric(.8, .4)])
            best = train_fight.train(arguments(root, epochs=2))  # Legacy Namespace has no added fields.
            run = json.loads((best.parent / "run.json").read_text())
            self.assertEqual(run["best_epoch"], 2)
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["improved_epochs"], [1, 2])
            self.assertEqual(json.loads(best.read_text())["val_metrics"]["loss"], .4)

    def test_default_output_directories_are_unique(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.manifest(root)
            self.mock_training(stack, [metric(), metric()] * 2)
            first = train_fight.train(arguments(root, output_dir=None, epochs=1))
            second = train_fight.train(arguments(root, output_dir=None, epochs=1))
            self.assertNotEqual(first.parent, second.parent)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())

    def test_accumulation_configuration_is_passed_only_to_training_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.manifest(root)
            _, _, _, epochs = self.mock_training(stack, [metric(), metric()])
            best = train_fight.train(arguments(root, epochs=1, accumulation_steps=4, freeze_batch_norm=True))
            run = json.loads((best.parent / "run.json").read_text())
            self.assertEqual(run["hyperparameters"]["effective_batch_size"], 8)
            self.assertEqual(run["hyperparameters"]["accumulation_steps"], 4)
            self.assertTrue(run["hyperparameters"]["freeze_batch_norm"])
            self.assertEqual(epochs.call_args_list[0].kwargs,
                             {"accumulation_steps": 4, "freeze_batch_norm": True})
            self.assertEqual(epochs.call_args_list[1].kwargs, {})

    def test_failure_marks_run_failed_and_does_not_create_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.manifest(root)
            self.mock_training(stack, [ValueError("non-finite training or validation loss")])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                train_fight.train(arguments(root))
            run = json.loads((root / "experiment/run.json").read_text())
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["error_type"], "ValueError")
            self.assertFalse((root / "experiment/best.pt").exists())

    def test_nonfinite_loss_is_rejected_before_backward(self):
        torch = types.ModuleType("torch")
        context = MagicMock()
        torch.enable_grad = MagicMock(return_value=context)
        loss = MagicMock()
        loss.detach.return_value.cpu.return_value = float("nan")
        frames, targets, model, optimizer = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                train_fight._run_epoch(model, [(frames, targets)], lambda *args: loss, "cpu", optimizer)
        loss.backward.assert_not_called()
        optimizer.step.assert_not_called()

    def test_weighted_loss_is_invariant_to_batch_boundaries(self):
        torch = types.ModuleType("torch")
        torch.inference_mode = MagicMock(side_effect=lambda: MagicMock())

        def evaluate_batches(batches):
            # Per-sample losses 2, 4, 6, with class weights .2, .2, .5.
            losses = [2.0, 4.0, 6.0]
            labels = [0, 0, 1]
            weights = [.2, .2, .5]
            loader, batch_losses, denominators, batch_logits = [], [], [], []
            for indices in batches:
                frames, targets = MagicMock(), MagicMock()
                frames.to.return_value = frames
                targets.to.return_value = targets
                targets.numel.return_value = len(indices)
                targets.detach.return_value.cpu.return_value.tolist.return_value = [labels[i] for i in indices]
                denominator = sum(weights[i] for i in indices)
                loss = MagicMock()
                loss.detach.return_value.cpu.return_value = sum(losses[i] * weights[i] for i in indices) / denominator
                weighted_targets = MagicMock()
                weighted_targets.sum.return_value.detach.return_value.cpu.return_value = denominator
                logits = MagicMock()
                logits.argmax.return_value.detach.return_value.cpu.return_value.tolist.return_value = [
                    labels[i] for i in indices
                ]
                loader.append((frames, targets))
                batch_losses.append(loss)
                denominators.append(weighted_targets)
                batch_logits.append(logits)
            criterion = MagicMock(side_effect=batch_losses)
            criterion.weight.__getitem__.side_effect = denominators
            model = MagicMock(side_effect=batch_logits)
            with patch.dict(sys.modules, {"torch": torch}):
                return train_fight._run_epoch(model, loader, criterion, "cpu", None).loss

        expected = (2 * .2 + 4 * .2 + 6 * .5) / .9
        self.assertAlmostEqual(evaluate_batches([[0, 1, 2]]), expected)
        self.assertAlmostEqual(evaluate_batches([[0, 1], [2]]), expected)
        self.assertAlmostEqual(evaluate_batches([[0], [1], [2]]), expected)


class AccumulationTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("optional torch training dependency is not installed") from None
        cls.torch = torch
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.original_threads)

    def test_accumulation_matches_whole_batch_updates_including_short_final_group(self):
        torch = self.torch
        for sample_count in (3, 11):
            for weighted in (False, True):
                with self.subTest(sample_count=sample_count, weighted=weighted):
                    torch.manual_seed(77)
                    expected = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(),
                                                   torch.nn.Linear(5, 2)).double()
                    actual = copy.deepcopy(expected)
                    features = torch.randn(sample_count, 3, dtype=torch.float64)
                    labels = torch.tensor(([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0])[:sample_count])
                    weight = torch.tensor([.3, 1.7], dtype=torch.float64) if weighted else None
                    criterion = torch.nn.CrossEntropyLoss(weight=weight)
                    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=.05,
                                                         momentum=.9, weight_decay=.01)
                    actual_optimizer = torch.optim.SGD(actual.parameters(), lr=.05,
                                                       momentum=.9, weight_decay=.01)
                    loss_sum, denominator_sum = 0.0, 0.0
                    for offset in range(0, sample_count, 8):
                        batch_features, batch_labels = features[offset:offset + 8], labels[offset:offset + 8]
                        expected_optimizer.zero_grad(set_to_none=True)
                        loss = criterion(expected(batch_features), batch_labels)
                        denominator = float(weight[batch_labels].sum()) if weighted else len(batch_labels)
                        loss_sum += float(loss.detach()) * denominator
                        denominator_sum += denominator
                        loss.backward()
                        expected_optimizer.step()
                    microbatches = [(features[offset:offset + 2], labels[offset:offset + 2])
                                    for offset in range(0, sample_count, 2)]
                    with patch.object(actual_optimizer, "step", wraps=actual_optimizer.step) as step:
                        result = train_fight._run_epoch(actual, microbatches, criterion, "cpu",
                                                       actual_optimizer, accumulation_steps=4)
                    self.assertEqual(step.call_count, (sample_count + 7) // 8)
                    self.assertAlmostEqual(result.loss, loss_sum / denominator_sum, places=12)
                    for expected_parameter, actual_parameter in zip(expected.parameters(), actual.parameters(),
                                                                    strict=True):
                        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=1e-10, atol=1e-12)

    def test_batch_norm_statistics_freeze_but_affine_parameters_still_learn(self):
        torch = self.torch
        torch.manual_seed(12)
        norm = torch.nn.BatchNorm1d(3)
        dropout = torch.nn.Dropout(.25)
        model = torch.nn.Sequential(norm, dropout, torch.nn.Linear(3, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        features = torch.randn(6, 3) + 4
        labels = torch.tensor([0, 1, 0, 0, 1, 0])
        original_mean, original_var = norm.running_mean.clone(), norm.running_var.clone()
        original_weight, original_bias = norm.weight.detach().clone(), norm.bias.detach().clone()
        original_batches = norm.num_batches_tracked.clone()
        batches = [(features[offset:offset + 2], labels[offset:offset + 2]) for offset in range(0, 6, 2)]
        for _ in range(2):
            train_fight._run_epoch(model, batches, torch.nn.CrossEntropyLoss(), "cpu", optimizer,
                                  accumulation_steps=2, freeze_batch_norm=True)
            self.assertTrue(model.training)
            self.assertTrue(dropout.training)
            self.assertFalse(norm.training)
            torch.testing.assert_close(norm.running_mean, original_mean, rtol=0, atol=0)
            torch.testing.assert_close(norm.running_var, original_var, rtol=0, atol=0)
            torch.testing.assert_close(norm.num_batches_tracked, original_batches, rtol=0, atol=0)
        self.assertTrue(norm.weight.requires_grad)
        self.assertTrue(norm.bias.requires_grad)
        self.assertFalse(torch.equal(norm.weight, original_weight))
        self.assertFalse(torch.equal(norm.bias, original_bias))
