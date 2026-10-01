from research_engine.controller import AutonomousResearchController


class FakeQueue:
    def __init__(self):
        self.store = None
        self.state = None
        self.created = []

    def run_state(self, run_id):
        if self.state is None:
            raise ValueError("run not found")
        return self.state

    def create_run(self, *, run_id, topic, policy=None, discovery_payload=None):
        self.created.append((run_id, topic, policy, discovery_payload))
        self.state = {
            "id": run_id,
            "topic": topic,
            "status": "WAITING_HUMAN",
            "current_phase": "HUMAN_REVIEW",
        }

    def recover_expired_leases(self):
        return []


class FakePlanner:
    def reconcile(self, run_id):
        return None


def test_controller_starts_new_run_and_returns_terminal_state(monkeypatch):
    queue = FakeQueue()
    controller = AutonomousResearchController(
        queue=queue,
        planner=FakePlanner(),
        workers=[],
        poll_seconds=0,
    )
    monkeypatch.setattr(controller, "_human_review", lambda run_id: {"status": "PENDING"})
    result = controller.run(topic="RF", run_id="run:1")
    assert result.status == "WAITING_HUMAN"
    assert result.cycles == 0
    assert result.human_review["status"] == "PENDING"
    assert queue.created[0][0:2] == ("run:1", "RF")


def test_controller_refuses_run_id_topic_collision(monkeypatch):
    queue = FakeQueue()
    queue.state = {
        "id": "run:1",
        "topic": "different topic",
        "status": "RUNNING",
        "current_phase": "DISCOVERY",
    }
    controller = AutonomousResearchController(
        queue=queue,
        planner=FakePlanner(),
        workers=[],
        poll_seconds=0,
    )
    try:
        controller.run(topic="RF", run_id="run:1", max_cycles=1)
    except ValueError as exc:
        assert "different topic" in str(exc)
    else:
        raise AssertionError("topic collision must fail")
