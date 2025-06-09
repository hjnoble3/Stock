import logging
import os
from typing import Optional

from dotenv import load_dotenv
from tastytrade import Session
from tastytrade.utils import TastytradeError

DOTENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))

active_session: Optional[Session] = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("__main__")


def session_login() -> Session:
    """Login to TastyTrade.
    Manages a module-level active session. If a valid session exists, it's reused.
    Loads environment variables from a .env file.
    Uses username and password for login.

    Requires TASTYWORKS_LOGIN and TASTYWORKS_PASSWORD to be set in the .env file or environment.

    Returns:
        Session: An authenticated TastyTrade session object.

    """
    global active_session  # Ensure we are working with the module-level active_session

    # Check if there's an existing, valid session
    if active_session:
        try:
            if active_session.validate():  # validate() can make an API call
                logger.info("Using existing active session.")
                return active_session
            logger.info("Existing session failed API validation. Will attempt new login.")
            active_session = None  # Mark for re-login
        except Exception as e:
            logger.info(f"Error validating existing session: {e}. Will attempt new login.")
            active_session = None  # Mark for re-login

    # If active_session is None at this point, we need to log in.
    if not active_session:
        logger.info("Attempting to establish a new session...")

    # Load environment variables from .env file.
    # override=True ensures that if .env is changed and this function is called again,
    # os.environ reflects the latest values from the file.
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    username = os.environ.get("TASTYWORKS_LOGIN")
    password = os.environ.get("TASTYWORKS_PASSWORD")

    if not username:
        raise ValueError(
            "TASTYWORKS_LOGIN must be set in .env file or environment variables.",
        )

    if not active_session:
        logger.info("Attempting login with username and password...")
        if not password:
            raise ValueError(
                "TASTYWORKS_PASSWORD must be set in .env file or environment variables for password login.",
            )

        try:
            active_session = Session(login=username, password=password, remember_me=False, is_test=False)
            logger.info("Login with username and password successful.")
        except TastytradeError as e:
            logger.error(f"Error logging in with username/password: {e}")
            raise  # Re-raise the error to signal login failure to the caller
        except Exception as e:  # Catch any other unexpected errors
            logger.error(f"Unexpected error during username/password login: {e}")
            raise

    if not active_session:
        # This line should ideally not be reached if errors above are re-raised.
        # However, as a final safeguard:
        raise TastytradeError("Failed to establish a session after all attempts.")
    return active_session


def destroy_session(session: Optional[Session]) -> None:
    """Destroys the given session."""
    if session:
        logger.info("Destroying session...")
        session.destroy()
    else:
        logger.info("No session to destroy.")


### Uncomment to test
if __name__ == "__main__":
    try:
        session = session_login()
        logger.info("Login successful!")
        # You can add more testing code here, e.g., accessing account information
        # destroy_session(session)
    except Exception as e:
        logger.error(f"Login failed: {e}")
