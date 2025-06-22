import os
import time
import json
import boto3
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging for the forwarder itself
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("log_forwarder")

# AWS CloudWatch configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
LOG_GROUP_NAME = os.getenv("LOG_GROUP_NAME", "/harmony-seeker/application")
LOG_STREAM_NAME = os.getenv(
    "LOG_STREAM_NAME", f"local-{datetime.now().strftime('%Y-%m-%d')}"
)

# Log file to monitor
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "app.log")

# State file to track the last read position
STATE_FILE = Path("log_forwarder/last_position.json")


def get_cloudwatch_client():
    """
    Create and return a CloudWatch Logs client

    Returns:
        boto3.client: CloudWatch Logs client
    """
    return boto3.client(
        "logs",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )


def ensure_log_group_exists(client):
    """
    Ensure the log group exists, create if it doesn't

    Args:
        client: CloudWatch Logs client
    """
    try:
        client.create_log_group(logGroupName=LOG_GROUP_NAME)
        logger.info(f"Created log group: {LOG_GROUP_NAME}")
    except client.exceptions.ResourceAlreadyExistsException:
        logger.info(f"Log group already exists: {LOG_GROUP_NAME}")


def ensure_log_stream_exists(client):
    """
    Ensure the log stream exists, create if it doesn't

    Args:
        client: CloudWatch Logs client
    """
    try:
        client.create_log_stream(
            logGroupName=LOG_GROUP_NAME, logStreamName=LOG_STREAM_NAME
        )
        logger.info(f"Created log stream: {LOG_STREAM_NAME}")
        return None  # No sequence token for a new stream
    except client.exceptions.ResourceAlreadyExistsException:
        logger.info(f"Log stream already exists: {LOG_STREAM_NAME}")

        # Get the sequence token for an existing stream
        response = client.describe_log_streams(
            logGroupName=LOG_GROUP_NAME, logStreamNamePrefix=LOG_STREAM_NAME
        )

        for stream in response.get("logStreams", []):
            if stream["logStreamName"] == LOG_STREAM_NAME:
                return stream.get("uploadSequenceToken")

        return None


def load_last_position():
    """
    Load the last read position from the state file

    Returns:
        int: Last read position in bytes
    """
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                return data.get("position", 0)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading state file: {e}")

    return 0


def save_last_position(position):
    """
    Save the last read position to the state file

    Args:
        position (int): Position in bytes
    """
    try:
        STATE_FILE.parent.mkdir(exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump({"position": position}, f)
    except IOError as e:
        logger.error(f"Error writing state file: {e}")


def read_new_logs(file_path, last_position):
    """
    Read new logs from the file starting from the last position

    Args:
        file_path (str): Path to the log file
        last_position (int): Last read position in bytes

    Returns:
        tuple: (new_logs, new_position)
    """
    try:
        with open(file_path, "r") as f:
            f.seek(last_position)
            new_logs = f.readlines()
            new_position = f.tell()
            return new_logs, new_position
    except FileNotFoundError:
        logger.error(f"Log file not found: {file_path}")
        return [], last_position
    except IOError as e:
        logger.error(f"Error reading log file: {e}")
        return [], last_position


def push_logs_to_cloudwatch(client, logs, sequence_token=None):
    """
    Push logs to CloudWatch

    Args:
        client: CloudWatch Logs client
        logs (list): List of log lines
        sequence_token (str): Sequence token for the log stream

    Returns:
        str: Next sequence token
    """
    if not logs:
        logger.info("No new logs to push")
        return sequence_token

    # Format the log events
    log_events = []
    timestamp = int(time.time() * 1000)  # Current time in milliseconds

    for i, log in enumerate(logs):
        log = log.strip()
        if log:  # Skip empty lines
            log_events.append(
                {
                    "timestamp": timestamp
                    + i,  # Ensure events have different timestamps
                    "message": log,
                }
            )

    if not log_events:
        return sequence_token

    # Push logs to CloudWatch
    try:
        kwargs = {
            "logGroupName": LOG_GROUP_NAME,
            "logStreamName": LOG_STREAM_NAME,
            "logEvents": log_events,
        }

        if sequence_token:
            kwargs["sequenceToken"] = sequence_token

        response = client.put_log_events(**kwargs)
        logger.info(f"Successfully pushed {len(log_events)} log events to CloudWatch")
        return response.get("nextSequenceToken")

    except client.exceptions.InvalidSequenceTokenException as e:
        # Handle invalid sequence token
        logger.warning(f"Invalid sequence token: {e}")
        # Extract the expected sequence token from the error message
        import re

        match = re.search(r"sequenceToken is: (\S+)", str(e))
        if match:
            correct_token = match.group(1)
            logger.info(f"Retrying with correct token: {correct_token}")
            kwargs["sequenceToken"] = correct_token
            response = client.put_log_events(**kwargs)
            return response.get("nextSequenceToken")
        return None

    except Exception as e:
        logger.error(f"Error pushing logs to CloudWatch: {e}")
        return sequence_token


def main():
    """Main function to monitor log file and push to CloudWatch"""
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        logger.error(
            "AWS credentials not found. Please set AWS_ACCESS_KEY and AWS_SECRET_KEY environment variables."
        )
        return

    logger.info("Starting CloudWatch log forwarder")
    logger.info(f"Monitoring log file: {LOG_FILE_PATH}")
    logger.info(f"Pushing to log group: {LOG_GROUP_NAME}, stream: {LOG_STREAM_NAME}")

    # Initialize CloudWatch client
    client = get_cloudwatch_client()

    # Ensure log group and stream exist
    ensure_log_group_exists(client)
    sequence_token = ensure_log_stream_exists(client)

    # Get the last read position
    last_position = load_last_position()
    logger.info(f"Starting from position: {last_position}")

    try:
        while True:
            # Read new logs
            new_logs, new_position = read_new_logs(LOG_FILE_PATH, last_position)

            # Push logs to CloudWatch
            if new_logs:
                logger.info(f"Found {len(new_logs)} new log lines")
                sequence_token = push_logs_to_cloudwatch(
                    client, new_logs, sequence_token
                )

                # Update the last read position
                last_position = new_position
                save_last_position(last_position)

            # Sleep before checking for new logs
            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("Log forwarder stopped")
        save_last_position(last_position)


if __name__ == "__main__":
    main()
