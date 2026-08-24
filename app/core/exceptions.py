from fastapi import status


class AnalysisError(Exception):
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class InvalidUUIDError(AnalysisError):
    def __init__(
        self, message: str = "Invalid contact_id format. Must be a valid UUID."
    ):
        super().__init__(
            message=message,
            code="INVALID_CONTACT_ID",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class AudioTooLargeError(AnalysisError):
    def __init__(self, message: str = "Audio file exceeds maximum allowed size."):
        super().__init__(
            message=message,
            code="AUDIO_TOO_LARGE",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )


class InvalidAudioError(AnalysisError):
    def __init__(self, message: str = "Invalid or undecodable audio file."):
        super().__init__(
            message=message,
            code="INVALID_AUDIO",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class ModelInferenceError(AnalysisError):
    def __init__(self, message: str = "Error occurred during model inference."):
        super().__init__(
            message=message,
            code="INFERENCE_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
