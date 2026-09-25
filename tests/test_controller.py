from pathlib import Path

from scripts import controller


def test_start_service_bootstraps_then_kickstarts_when_unloaded(tmp_path, monkeypatch):
    service = controller.Service("com.example.ethladder", "example.plist")
    source = tmp_path / "source.plist"
    installed = tmp_path / "installed.plist"
    source.write_text("<plist/>", encoding="utf-8")
    calls = []

    monkeypatch.setattr(controller.Service, "source_plist", property(lambda self: source))
    monkeypatch.setattr(controller.Service, "installed_plist", property(lambda self: installed))
    monkeypatch.setattr(controller, "is_loaded", lambda _: False)
    monkeypatch.setattr(
        controller,
        "run",
        lambda *args, **kwargs: calls.append(args),
    )

    controller.start_service(service)

    assert installed.read_text(encoding="utf-8") == "<plist/>"
    assert calls == [
        ("launchctl", "bootstrap", controller.DOMAIN, str(installed)),
        ("launchctl", "kickstart", "-k", f"{controller.DOMAIN}/{service.label}"),
    ]
