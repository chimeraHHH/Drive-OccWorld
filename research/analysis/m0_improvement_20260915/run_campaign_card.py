"""One bounded server-side card of the frozen observation-memory campaign.

anchor: extract train512 -> wait development200 -> train persistent2.
control: extract development200 -> wait train512 -> train native1 -> rolling2.

Run under job_runner.py. subprocess.run inherits this controller's environment,
cwd and PGID (no shell, no new session), so the parent can clean only its owned
process group. Scientific stages never retry. A role directory is claimed once;
only explicitly requested, fully bound COMPLETE caches may be reused.

Paths: --root/cards/<role> holds controller receipts, --root/runs/<arm> holds
learner artifacts, --cache-root/{train,development} holds native state caches.
The same absolute --deadline-unix must be passed to both cards. A monotonic
counter also bounds the initial remaining duration against wall-clock rollback.
No torch/MMCV import is needed by this controller.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


M0_SHA256 = "0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc"
MODEL_SOURCE_SHA256 = "67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5"
ROLES = {"anchor": ("train", "development", ("persistent2",)),
         "control": ("development", "train", ("native1", "rolling2"))}
IDENTITY_KEYS = ("sample_token", "scene_token", "official_index", "split")


class CampaignStop(RuntimeError):
    pass


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise CampaignStop(message)


def durable_json(path, value, exclusive=False):
    """Fsync bytes and directory; final receipts are atomically create-only."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()) + "." + str(time.time_ns()))
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if exclusive:
            # Atomic creation without replacing an existing completed artifact.
            os.link(str(temporary), str(path))
            temporary.unlink()
        else:
            os.replace(str(temporary), str(path))
        descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


class Card:
    def __init__(self, args):
        self.a = args
        self.package = Path(__file__).resolve().parent
        self.root = Path(args.root).resolve()
        self.cache_root = Path(args.cache_root).resolve()
        self.directory = self.root / "cards" / args.role
        # mkdir is the lifetime claim; completed or failed cards cannot restart.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = (self.directory / "owner.lock").open("x")
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.started = time.monotonic()
        self.monotonic_deadline = self.started + max(0., args.deadline_unix-time.time())
        self.protocol_path = Path(args.protocol).resolve()
        self.selection_path = self.protocol_path.with_name("selection_v1.json")
        self.config = Path(args.config).resolve()
        self.checkpoint = Path(args.checkpoint).resolve()
        self.peer = "control" if args.role == "anchor" else "anchor"
        self.state = dict(schema="m0-memory-campaign-card-v1", status="VALIDATING",
                          role=args.role, started_utc=utc(), pid=os.getpid(),
                          pgid=os.getpgid(0), root=str(self.root),
                          cache_root=str(self.cache_root), package=str(self.package),
                          deadline_unix=args.deadline_unix, completed_stages=[])
        self.protocol = None
        self.expected_rows = None
        self.contract = None
        durable_json(self.directory / "state.json", self.state)

    def update(self, **fields):
        self.state.update(fields)
        self.state.update(updated_utc=utc(), elapsed_seconds=time.monotonic()-self.started)
        durable_json(self.directory / "state.json", self.state)

    def remaining(self):
        return min(self.a.deadline_unix-time.time(), self.monotonic_deadline-time.monotonic())

    def check_stop(self):
        require(self.remaining() > 0, "GLOBAL_DEADLINE")
        peer_failure = self.root / "cards" / self.peer / "failed.json"
        if peer_failure.exists():
            raise CampaignStop("PEER_FAILED: " + str(peer_failure))
        peer_state = self.root / "cards" / self.peer / "state.json"
        if peer_state.is_file():
            value = read(peer_state)
            require(value.get("deadline_unix") == self.a.deadline_unix,
                    "Cards do not share the same global deadline")
            require(value.get("cache_root") == str(self.cache_root),
                    "Cards do not share the same cache root")
            require(value.get("status") not in ("FAILED", "INTERRUPTED", "DEADLINE"),
                    "PEER_TERMINAL_FAILURE: " + str(peer_state))
        peer_manifest = self.root / "cards" / self.peer / "manifest.json"
        if peer_manifest.is_file():
            require(read(peer_manifest)["contract"]["protocol_sha256"] == self.a.protocol_sha256,
                    "Peer card is using a different frozen protocol")
        for split in ("train", "development"):
            if (self.cache_root / split / "failed.json").exists():
                raise CampaignStop("CACHE_FAILED: " + split)

    def validate_contract(self, initial=False):
        self.check_stop()
        require(sha(self.protocol_path) == self.a.protocol_sha256, "Protocol hash mismatch")
        protocol = read(self.protocol_path)
        require(sha(self.config) == protocol["config_sha256"], "Config hash mismatch")
        require(sha(self.selection_path) == protocol["selection_sha256"], "Selection hash mismatch")
        source_hashes = protocol["source_sha256"]
        required = {"run_campaign_card.py", "native_state_cache.py", "memory_experiment.py", "observation_memory.py"}
        require(required.issubset(source_hashes), "Protocol must bind controller, extractor, learner and adapter")
        for name, expected in source_hashes.items():
            require(Path(name).name == name, "Source keys must be package basenames")
            require(sha(self.package / name) == expected, "Frozen source changed: " + name)
        runtime_hashes = protocol["runtime_source_sha256"]
        require(isinstance(runtime_hashes, dict) and runtime_hashes,
                "Frozen native runtime source contract is required")
        for filename, expected in runtime_hashes.items():
            require(Path(filename).is_absolute(), "Runtime source keys must be server absolute paths")
            require(sha(filename) == expected, "Frozen native runtime source changed: " + filename)
        hp = protocol["training"]
        require(hp["seed"] == 11 and hp["scope"] == "whole_future_pred_head", "Training scope/seed mismatch")
        require(hp["train_samples"] == 512 and hp["development_samples"] == 200, "Unexpected sample counts")
        require(isinstance(hp["passes"], int) and hp["passes"] > 0 and
                isinstance(hp["accumulate"], int) and hp["accumulate"] > 0 and 512 % hp["accumulate"] == 0,
                "Invalid fixed training budget")
        # Hash the checkpoint at entry and after each scientific stage; not while
        # waiting every 30s. No local hash memoization can conceal stage mutation.
        if initial:
            require(sha(self.checkpoint) == M0_SHA256, "Native M0 checkpoint mismatch")
        selection = read(self.selection_path)
        rows = {split: [r for r in selection["records"] if r["split"] == split]
                for split in ("train", "development")}
        require(len(rows["train"]) == 512 and len(rows["development"]) == 200,
                "Selection must contain train512 and development200")
        require(len(selection["records"]) == 712, "Unexpected additional selected roles")
        for split, expected in rows.items():
            require(len({r["sample_token"] for r in expected}) == len(expected), "Duplicate tokens: " + split)
            require(len({r["official_index"] for r in expected}) == len(expected), "Duplicate official indices: " + split)
        require(not {r["scene_token"] for r in rows["train"]} &
                {r["scene_token"] for r in rows["development"]}, "Train/development scene overlap")
        if initial:
            self.protocol, self.expected_rows = protocol, rows
            self.contract = dict(protocol_sha256=self.a.protocol_sha256,
                                 config_sha256=protocol["config_sha256"],
                                 selection_sha256=protocol["selection_sha256"],
                                 source_sha256=source_hashes,
                                 runtime_source_sha256=runtime_hashes,
                                 checkpoint_sha256=M0_SHA256,
                                 extract_max_seconds=self.a.extract_max_seconds,
                                 extract_max_bytes_per_split=self.a.extract_max_bytes,
                                 deadline_unix=self.a.deadline_unix,
                                 reuse_complete_caches=self.a.reuse_complete_caches)
            durable_json(self.directory / "manifest.json", dict(self.state, contract=self.contract), exclusive=True)
        self.check_stop()

    def verify_cache(self, split):
        cache = self.cache_root / split
        complete = read(cache / "complete.json")
        index_hash = sha(cache / "index.json")
        index = read(cache / "index.json")
        expected = self.expected_rows[split]
        require(complete["status"] == "COMPLETE_NATIVE_STATE_CACHE", "Cache not successfully complete")
        require(complete["index_sha256"] == index_hash, "Cache index hash mismatch")
        require(index["schema"] == "m0-native-state-cache-v1" and index["status"] == "COMPLETE", "Invalid cache schema/state")
        require(index["split"] == split and index["seed"] == 11, "Cache role/seed mismatch")
        require(index["selection_sha256"] == self.contract["selection_sha256"], "Cache selection changed")
        require(index["config_sha256"] == self.contract["config_sha256"], "Cache config changed")
        require(index["script_sha256"] == self.contract["source_sha256"]["native_state_cache.py"], "Cache extractor changed")
        native = index["native_model"]
        require(native["checkpoint_sha256"] == M0_SHA256 and native["source_model_sha256"] == MODEL_SOURCE_SHA256,
                "Cache does not derive from frozen native M0")
        require(native["config_sha256"] == self.contract["config_sha256"], "Native cache config provenance mismatch")
        require(index["future_occupancy_is_input"] is False and index["inputs_targets_separate"] is True,
                "Cache input/label separation changed")
        require(index["future_ego_action_conditioned"] is True and index["extraction_model_mode"] == "eval_no_grad",
                "Cache native condition/mode changed")
        require(complete["optimizer_steps"] == 0 and complete["native_radar_unchanged"] is True
                and complete["all_samples_bitwise_replayed"] is True, "Native replay precondition failed")
        actual = index["records"]
        require(complete["samples"] == len(expected) == len(actual), "Incomplete sample count")
        require([[r[k] for k in IDENTITY_KEYS] for r in actual] ==
                [[r[k] for k in IDENTITY_KEYS] for r in expected], "Cache identities/order changed")
        require(all(r["parity"]["status"] == "PASS" for r in actual), "Cache per-sample parity failure")
        require(complete["cache_bytes"] == index["cache_bytes"] and
                index["cache_bytes"] <= self.a.extract_max_bytes, "Cache byte count exceeds frozen CLI cap")
        # Only metadata is read here. Actual tensors are checked by load_sample
        # against the bound record SHA before deserialization in the learner.
        return dict(split=split, directory=str(cache), samples=len(actual),
                    index_sha256=index_hash, complete_sha256=sha(cache / "complete.json"),
                    cache_bytes=index["cache_bytes"], payload_verification="Per-file SHA at learner load; not re-read by controller")

    def run_stage(self, name, argv, verify):
        self.validate_contract()
        stage = self.directory / "stages" / name
        stage.mkdir(parents=True, exist_ok=False)
        launch = dict(stage=name, argv=argv, cwd=os.getcwd(), pgid=os.getpgid(0),
                      started_utc=utc(), deadline_unix=self.a.deadline_unix,
                      timeout_seconds=self.remaining(), contract=self.contract,
                      shell=False, inherited_environment=True, new_session=False)
        durable_json(stage / "launch.json", launch, exclusive=True)
        self.update(status="RUNNING_STAGE", stage=name, argv=argv)
        began = time.monotonic()
        try:
            with (stage / "stdout.log").open("x") as log:
                try:
                    proc = subprocess.run(argv, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT,
                                          timeout=max(.001, self.remaining()), check=False)
                finally:
                    log.flush(); os.fsync(log.fileno())
            require(proc.returncode == 0, "Stage nonzero returncode: %s=%s" % (name, proc.returncode))
            self.validate_contract()
            require(sha(self.checkpoint) == M0_SHA256, "Native checkpoint changed during stage")
            result = verify()
            receipt = dict(status="PASS", stage=name, returncode=proc.returncode,
                           started_utc=launch["started_utc"], finished_utc=utc(),
                           seconds=time.monotonic()-began, argv=argv,
                           stdout_sha256=sha(stage / "stdout.log"), artifacts=result,
                           contract=self.contract, automatic_retry=False)
            durable_json(stage / "complete.json", receipt, exclusive=True)
            self.state["completed_stages"].append(dict(stage=name,receipt_sha256=sha(stage / "complete.json")))
            self.update(status="BETWEEN_STAGES", stage=name)
            return result
        except BaseException as exc:
            durable_json(stage / "failed.json", dict(status="FAILED", stage=name,
                         error=repr(exc), seconds=time.monotonic()-began,
                         stdout_sha256=sha(stage / "stdout.log") if (stage / "stdout.log").is_file() else None,
                         automatic_retry=False), exclusive=True)
            raise

    def extract(self, split):
        target = self.cache_root / split
        if target.exists():
            require(self.a.reuse_complete_caches, "Existing cache requires --reuse-complete-caches")
            self.check_stop()
            require((target / "complete.json").is_file(), "Existing incomplete cache cannot resume")
            result = self.verify_cache(split)
            durable_json(self.directory / ("reuse_" + split + ".json"),
                         dict(status="VERIFIED_CACHE_REUSE", artifact=result, contract=self.contract), exclusive=True)
            return result
        seconds = max(1, int(min(self.a.extract_max_seconds, self.remaining())))
        argv = [sys.executable, "-B", str(self.package / "native_state_cache.py"),
                "--mode", "extract", "--config", str(self.config), "--checkpoint", str(self.checkpoint),
                "--selection", str(self.selection_path), "--split", split,
                "--out", str(target), "--device", "cuda:0", "--max-seconds", str(seconds),
                "--max-bytes", str(self.a.extract_max_bytes)]
        return self.run_stage("extract_" + split, argv, lambda: self.verify_cache(split))

    def wait_cache(self, split):
        complete = self.cache_root / split / "complete.json"
        self.update(status="WAITING_FOR_CACHE", waiting_for=split)
        while not complete.is_file():
            self.check_stop()
            self.update(status="WAITING_FOR_CACHE", waiting_for=split,
                        remaining_seconds=self.remaining())
            time.sleep(min(30., max(.001, self.remaining())))
        self.check_stop()
        self.validate_contract()
        result = self.verify_cache(split)
        durable_json(self.directory / ("dependency_" + split + ".json"),
                     dict(status="VERIFIED_PEER_CACHE", artifact=result, contract=self.contract), exclusive=True)
        return result

    def verify_training(self, arm, caches):
        out = self.root / "runs" / arm
        complete, manifest = read(out / "complete.json"), read(out / "manifest.json")
        trained = read(out / "training_complete.json")
        hp = self.protocol["training"]
        updates, examples = hp["passes"]*512//hp["accumulate"], hp["passes"]*512
        require(complete["status"] == "TRAINED_AND_DEVELOPMENT_EVALUATED", "Arm did not finish")
        require(trained["status"] == "TRAINED_FIXED_FINAL", "Wrong checkpoint selection rule")
        for row in (complete, trained):
            require(row["updates"] == updates and row["examples"] == examples, "Training budget mismatch")
            require(row["checkpoint_sha256"] == sha(out / "latest.pth"), "Final checkpoint hash mismatch")
        require(manifest["arm"] == arm and manifest["seed"] == 11, "Training arm/seed mismatch")
        require(manifest["protocol_sha256"] == self.a.protocol_sha256 and manifest["m0_sha256"] == M0_SHA256,
                "Training source checkpoint/protocol mismatch")
        require(manifest["train_index_sha256"] == caches["train"]["index_sha256"] and
                manifest["dev_index_sha256"] == caches["development"]["index_sha256"], "Training cache changed")
        for name, expected in manifest["source_sha256"].items():
            require(expected == self.contract["source_sha256"][name], "Training source provenance mismatch")
        require(manifest["migration"]["mode"] == arm, "Wrong installed memory policy")
        require(complete["evaluation"]["samples"] == 200 and
                complete["evaluation"]["sha256"] == sha(out / "development_records.jsonl"), "Evaluation receipt mismatch")
        eval_rows = [json.loads(line) for line in (out / "development_records.jsonl").read_text().splitlines() if line.strip()]
        require([[r[k] for k in ("sample_token", "scene_token")] for r in eval_rows] ==
                [[r[k] for k in ("sample_token", "scene_token")] for r in self.expected_rows["development"]],
                "Evaluation identities/order mismatch")
        for split in caches:
            require(self.verify_cache(split) == caches[split], "Cache metadata changed during training")
        return dict(arm=arm,updates=updates,examples=examples,directory=str(out),
                    file_sha256={name:sha(out / name) for name in
                                 ("manifest.json", "training_complete.json", "complete.json", "latest.pth",
                                  "training.jsonl", "development_records.jsonl")})

    def run(self):
        self.validate_contract(initial=True)
        owned_split, peer_split, arms = ROLES[self.a.role]
        (self.root / "runs").mkdir(exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        for arm in arms:
            require(not (self.root / "runs" / arm).exists(), "Existing arm artifact must not be overwritten: " + arm)
        caches = {owned_split: self.extract(owned_split), peer_split: self.wait_cache(peer_split)}
        for arm in arms:
            argv = [sys.executable, "-B", str(self.package / "memory_experiment.py"),
                    "--mode", "train", "--config", str(self.config), "--checkpoint", str(self.checkpoint),
                    "--train-cache", str(self.cache_root / "train"),
                    "--dev-cache", str(self.cache_root / "development"),
                    "--protocol", str(self.protocol_path), "--arm", arm,
                    "--out", str(self.root / "runs" / arm)]
            self.run_stage("train_" + arm, argv, lambda arm=arm: self.verify_training(arm, caches))
        self.update(status="COMPLETE", completed_utc=utc(), caches=caches)
        durable_json(self.directory / "complete.json", self.state, exclusive=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--extract-max-seconds", type=int, required=True)
    parser.add_argument("--extract-max-bytes", type=int, required=True)
    parser.add_argument("--reuse-complete-caches", action="store_true")
    args = parser.parse_args(argv)
    require(math.isfinite(args.deadline_unix), "Invalid deadline")
    require(args.extract_max_seconds > 0 and args.extract_max_bytes > 0, "Positive extraction bounds required")
    require(len(args.protocol_sha256) == 64 and all(c in "0123456789abcdef" for c in args.protocol_sha256),
            "Expected lowercase protocol SHA256")
    card = Card(args)
    def stopped(number, _frame):
        raise CampaignStop("SIGNAL_" + str(number))
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        card.run()
        print(json.dumps(dict(status="COMPLETE",role=args.role,receipt=str(card.directory / "complete.json"))), flush=True)
        return 0
    except BaseException as exc:
        card.update(status="FAILED",error=repr(exc),automatic_retry=False)
        durable_json(card.directory / "failed.json", card.state, exclusive=True)
        traceback.print_exc()
        return 1
    finally:
        card.lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
