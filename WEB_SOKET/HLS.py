"""
This script provides functionality to generate an HLS (HTTP Live Streaming) URL for a Kinesis Video Stream 
using AWS SDK for Python (Boto3). The script includes the following:
Functions:
- generate_hls_url(stream_name, playlist_expiration=36000): 
    Generates an HLS streaming session URL for a specified Kinesis Video Stream. 
    It retrieves the data endpoint for the stream, creates a Kinesis Video Media client, 
    and requests the HLS streaming session URL.
Parameters:
- stream_name (str): The name of the Kinesis Video Stream for which the HLS URL is to be generated.
- playlist_expiration (int): The expiration time for the HLS playlist in seconds. Defaults to 36000 seconds (10 hours).
Returns:
- str: The HLS streaming session URL if successful, or None if an error occurs.
Exceptions Handled:
- NoCredentialsError: Raised when AWS credentials are not found.
- ClientError: Raised for other client-related errors, with the error message printed.
Usage:
- Replace 'YourStreamName' in the `__main__` block with the name of your Kinesis Video Stream.
- Run the script to generate and print the HLS URL for the specified stream.
Dependencies:
- boto3: AWS SDK for Python.
- botocore.exceptions: For handling AWS-related exceptions.
Note:
- Ensure that AWS credentials are configured properly in your environment.
- Replace the region name in the `kvs_client` initialization with the appropriate AWS region for your stream.
"""
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Initialize the Kinesis Video Streams client
kvs_client = boto3.client('kinesisvideo', region_name='us-east-1')  # Replace with your region

def generate_hls_url(stream_name, playlist_expiration=36000):
    """
    Generate an HLS streaming session URL for a Kinesis Video Stream.

    :param stream_name: Name of the Kinesis Video Stream.
    :param playlist_expiration: Expiration time in seconds (default: 3600 seconds = 1 hour).
    :return: HLS URL as a string.
    """
    try:
        # Get the DataEndpoint for the Kinesis Video Stream
        data_endpoint_response = kvs_client.get_data_endpoint(
            StreamName=stream_name,
            APIName='GET_HLS_STREAMING_SESSION_URL'
        )
        data_endpoint = data_endpoint_response['DataEndpoint']

        # Create a Kinesis Video Media client using the DataEndpoint
        kvam_client = boto3.client('kinesis-video-archived-media', endpoint_url=data_endpoint)

        # Generate the HLS Streaming Session URL
        hls_url_response = kvam_client.get_hls_streaming_session_url(
            StreamName=stream_name,
            PlaybackMode='LIVE',  # Use 'ON_DEMAND' for archived streams
            ContainerFormat='MPEG_TS',
            DisplayFragmentTimestamp='ALWAYS',
            Expires=playlist_expiration
        )

        return hls_url_response['HLSStreamingSessionURL']
    
    except NoCredentialsError:
        print("AWS credentials not found.")
        return None
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None

if __name__ == "__main__":
    # Replace 'YourStreamName' with the name of your Kinesis Video Stream
    stream_name = 'Camer_121'

    # Generate the HLS URL
    hls_url = generate_hls_url(stream_name)
    if hls_url:
        print(f"HLS URL: {hls_url}")