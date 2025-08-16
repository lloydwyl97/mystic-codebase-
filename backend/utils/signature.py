import base64
import hashlib
import hmac

from enhanced_logging import log_operation_performance


@log_operation_performance("create_signature")
def create_signature(api_secret: str, message: str) -> str:
    """
    Create a signature for Coinbase API requests

    Args:
        api_secret: The API secret key (base64 encoded)
        message: The message to sign

    Returns:
        The signature as a base64 encoded string
    """
    try:
        # Decode the API secret from base64
        secret_bytes = base64.b64decode(api_secret)

        # Create HMAC signature
        hmac_obj = hmac.new(
            key=secret_bytes,
            msg=message.encode("utf-8"),
            digestmod=hashlib.sha256,
        )
        return base64.b64encode(hmac_obj.digest()).decode("utf-8")
    except Exception as e:
        # Handle potential errors with base64 decoding
        raise ValueError(f"Error creating signature: {str(e)}")


class SignatureManager:
    pass


