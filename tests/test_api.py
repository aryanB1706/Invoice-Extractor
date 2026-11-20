"""API smoke tests."""

from fastapi.testclient import TestClient

from invoiceiq.api.app import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_extract_endpoint_is_reserved() -> None:
    response = client.post(
        "/api/v1/extract",
        files={"file": ("invoice.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 501
