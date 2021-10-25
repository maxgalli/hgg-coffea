import hgg_coffea as m


def test_version():
    assert m.__version__


def test_tools_subpackage():
    from hgg_coffea.tools import chained_quantile  # noqa
    from hgg_coffea.tools import diphoton_mva  # noqa
    from hgg_coffea.tools import xgb_loader  # noqa


def test_workflows_subpackage():
    from hgg_coffea.workflows import base  # noqa
    from hgg_coffea.workflows import dystudies  # noqa
