import pytest

from agentforge.eval.tau2_runner import Tau2BenchNotConfigured, run_tau2_eval


class TestTau2RunnerNotYetImplemented:
    def test_raises_with_actionable_message(self) -> None:
        with pytest.raises(Tau2BenchNotConfigured, match="real spike"):
            run_tau2_eval(
                repo="https://github.com/sierra-research/tau2-bench",
                adapter_dir="/adapters/run1",
                base_model="google/gemma-4-12B-it",
                domains=None,
                run_name="test",
            )
