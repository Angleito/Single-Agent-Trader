#!/usr/bin/env python3
"""
CDP Key Extractor

This script extracts CDP API key information from a Coinbase CDP API key JSON file
and provides the values needed for environment variables.

Usage:
    python scripts/extract_cdp_keys.py /path/to/cdp-api-key.json

The script will output the CDP_API_KEY_NAME and CDP_PRIVATE_KEY values
that you can copy to your .env file.

Expected JSON format:
{
   "name": "organizations/.../apiKeys/...",
   "privateKey": "-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----\n"
}
"""

import json
import sys
from pathlib import Path


def extract_cdp_keys(json_file_path: str) -> dict:
    """
    Extract CDP API key information from JSON file.

    Args:
        json_file_path: Path to the CDP API key JSON file

    Returns:
        Dictionary with extracted key information

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        ValueError: If the JSON format is invalid
        KeyError: If required fields are missing
    """
    file_path = Path(json_file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CDP API key file not found: {json_file_path}")

    if not file_path.suffix.lower() == ".json":
        raise ValueError(f"File must be a JSON file: {json_file_path}")

    try:
        with open(file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # Extract required fields
    if "name" not in data:
        raise KeyError("Missing 'name' field in CDP API key file")

    if "privateKey" not in data:
        raise KeyError("Missing 'privateKey' field in CDP API key file")

    api_key_name = data["name"]
    private_key = data["privateKey"]

    # Validate key name format
    if not api_key_name.startswith("organizations/"):
        raise ValueError(f"Invalid API key name format: {api_key_name}")

    # Validate private key format
    if not private_key.startswith("-----BEGIN EC PRIVATE KEY-----"):
        raise ValueError(
            "Private key must be in PEM format starting with '-----BEGIN EC PRIVATE KEY-----'"
        )

    if not private_key.strip().endswith("-----END EC PRIVATE KEY-----"):
        raise ValueError("Private key must end with '-----END EC PRIVATE KEY-----'")

    return {
        "api_key_name": api_key_name,
        "private_key": private_key,
        "organization_id": api_key_name.split("/")[1] if "/" in api_key_name else None,
        "key_id": api_key_name.split("/")[-1] if "/" in api_key_name else None,
    }


def format_env_output(extracted_keys: dict) -> str:
    """
    Format the extracted keys for .env file usage.

    Args:
        extracted_keys: Dictionary with extracted key information

    Returns:
        Formatted string for .env file
    """
    # Escape the private key for shell/env file usage
    private_key_escaped = extracted_keys["private_key"].replace("\n", "\\n")

    output = []
    output.append("# CDP API Keys - Copy these to your .env file")
    output.append("#" + "=" * 60)
    output.append("")
    output.append(f"CDP_API_KEY_NAME={extracted_keys['api_key_name']}")
    output.append(f'CDP_PRIVATE_KEY="{private_key_escaped}"')
    output.append("")
    output.append("# Additional Information:")
    if extracted_keys.get("organization_id"):
        output.append(f"# Organization ID: {extracted_keys['organization_id']}")
    if extracted_keys.get("key_id"):
        output.append(f"# Key ID: {extracted_keys['key_id']}")
    output.append("# Authentication Method: CDP")
    output.append("")
    output.append("# IMPORTANT:")
    output.append("# - Make sure to comment out or remove legacy Coinbase credentials")
    output.append("# - Do not use both legacy and CDP credentials simultaneously")
    output.append(
        "# - Keep these credentials secure and never commit them to version control"
    )

    return "\n".join(output)


def main():
    """Main function to handle command line usage."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/extract_cdp_keys.py /path/to/cdp-api-key.json")
        print("")
        print(
            "This script extracts CDP API key information from a Coinbase CDP API key JSON file."
        )
        print("The output can be copied to your .env file.")
        print("")
        print("Expected JSON format:")
        print("{")
        print('   "name": "organizations/.../apiKeys/...",')
        print(
            '   "privateKey": "-----BEGIN EC PRIVATE KEY-----\\n...\\n-----END EC PRIVATE KEY-----\\n"'
        )
        print("}")
        sys.exit(1)

    json_file_path = sys.argv[1]

    try:
        # Extract the keys
        extracted_keys = extract_cdp_keys(json_file_path)

        # Format and display the output
        env_output = format_env_output(extracted_keys)
        print(env_output)

        # Offer to save to file
        print("\n" + "=" * 70)
        save_choice = input("Save this output to a file? (y/N): ").strip().lower()

        if save_choice in ["y", "yes"]:
            output_file = input(
                "Enter output filename (default: cdp_env_vars.txt): "
            ).strip()
            if not output_file:
                output_file = "cdp_env_vars.txt"

            with open(output_file, "w") as f:
                f.write(env_output)

            print(f"Environment variables saved to: {output_file}")
            print("You can now copy the contents to your .env file.")

        print("\nExtraction completed successfully!")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except (ValueError, KeyError) as e:
        print(f"ERROR: {e}")
        print("\nPlease check that your JSON file has the correct format:")
        print("{")
        print('   "name": "organizations/.../apiKeys/...",')
        print(
            '   "privateKey": "-----BEGIN EC PRIVATE KEY-----\\n...\\n-----END EC PRIVATE KEY-----\\n"'
        )
        print("}")
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
