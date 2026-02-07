#!/usr/bin/env python3
"""
Manually download NLTK resources.
This script helps download required NLTK resources when automatic download fails.
"""
import sys
import argparse

def download_nltk_resource(resource_name: str, quiet: bool = False):
    """Download a specific NLTK resource."""
    try:
        import nltk
        
        if not quiet:
            print(f"Downloading NLTK resource: {resource_name}")
            print(f"  This may take a while...")
        
        nltk.download(resource_name, quiet=quiet)
        
        if not quiet:
            print(f"  ✓ Successfully downloaded {resource_name}")
        return True
        
    except Exception as e:
        if not quiet:
            print(f"  ✗ Failed to download {resource_name}: {e}")
        return False

def check_nltk_resource(resource_name: str):
    """Check if an NLTK resource is already available."""
    try:
        import nltk
        
        # Try to find the resource based on resource name
        # punkt_tab is located at tokenizers/punkt_tab
        if resource_name == "punkt_tab":
            nltk.data.find('tokenizers/punkt_tab')
        else:
            # For other resources, try common paths
            nltk.data.find(resource_name)
        
        print(f"  ✓ Resource '{resource_name}' is already available")
        return True
    except LookupError:
        print(f"  ✗ Resource '{resource_name}' is not available")
        return False
    except Exception as e:
        print(f"  ? Could not check resource: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Manually download NLTK resources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download punkt_tab resource
  python scripts/download_nltk_manually.py punkt_tab

  # Check if resource exists
  python scripts/download_nltk_manually.py --check punkt_tab

  # Download all required resources
  python scripts/download_nltk_manually.py --all
        """
    )
    parser.add_argument(
        "resource",
        nargs="?",
        default=None,
        help="NLTK resource name to download (e.g., punkt_tab)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if resource exists instead of downloading"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all required resources"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    # Required resources for this project
    required_resources = ["punkt_tab"]
    
    if args.all:
        # Download all required resources
        success_count = 0
        for resource in required_resources:
            if args.check:
                if check_nltk_resource(resource):
                    success_count += 1
            else:
                if download_nltk_resource(resource, quiet=args.quiet):
                    success_count += 1
        
        if not args.quiet:
            print(f"\nSummary: {success_count}/{len(required_resources)} resources processed successfully")
        return 0 if success_count == len(required_resources) else 1
    
    elif args.resource:
        # Download specific resource
        if args.check:
            result = check_nltk_resource(args.resource)
            return 0 if result else 1
        else:
            result = download_nltk_resource(args.resource, quiet=args.quiet)
            return 0 if result else 1
    
    else:
        # Default: download all required resources
        if not args.quiet:
            print("No resource specified. Downloading all required resources...")
            print(f"Required resources: {', '.join(required_resources)}")
        
        success_count = 0
        for resource in required_resources:
            if download_nltk_resource(resource, quiet=args.quiet):
                success_count += 1
        
        if not args.quiet:
            print(f"\nSummary: {success_count}/{len(required_resources)} resources downloaded successfully")
        return 0 if success_count == len(required_resources) else 1

if __name__ == "__main__":
    sys.exit(main())

