"""Dataset-level evaluation runner."""

from invoiceiq.schemas import Invoice


def evaluate_dataset(expected: list[Invoice], predicted: list[Invoice]) -> dict[str, float]:
    """Aggregate field accuracy across a labelled dataset.

    Implementation is intentionally deferred to the evaluation milestone.
    """
    raise NotImplementedError("Dataset evaluation will be implemented in a later step.")
