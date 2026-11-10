"""Optional AIS screening workbench. Importing this package does not load ML models."""


def attach(app):
    """Mount the isolated plugin once, without changing existing routes or storage."""
    if getattr(app.state, "anomaly_workbench_attached", False):
        return
    from .live_service import router, install_catalog_entry

    app.include_router(router)
    install_catalog_entry(app)
    app.state.anomaly_workbench_attached = True


__all__ = ["attach"]
