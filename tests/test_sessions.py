import importlib.machinery
import pathlib
import types

path = pathlib.Path(__file__).resolve().parents[1] / "codefull"
loader = importlib.machinery.SourceFileLoader("codefull_module", str(path))
codefull = types.ModuleType("codefull_module")
loader.exec_module(codefull)

start_session = codefull.start_session
resume_session = codefull.resume_session
terminate_session = codefull.terminate_session


def test_variable_reuse_across_turns():
    session = start_session()
    ctx = session.context
    ctx.add_variable('a', 1)
    sid = session.id

    resumed = resume_session(sid)
    assert resumed is session
    resumed.context.add_variable('b', resumed.context.get_variable('a') + 1)
    assert resumed.context.get_variable('b') == 2
    terminate_session(sid)


def test_stepwise_refinement_with_undo():
    session = start_session()
    ctx = session.context
    ctx.add_variable('x', 1)
    ctx.save_state()
    ctx.add_variable('x', 2)
    assert ctx.get_variable('x') == 2
    ctx.undo()
    assert ctx.get_variable('x') == 1
    terminate_session(session.id)
