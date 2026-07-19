#!/usr/bin/env python3
"""Test script to verify the rendering fix."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'deep_agent'))

from cli import extract_content_from_result


def test_extract_content():
    """Test extraction of content from various result formats."""
    
    # Test case 1: List of dict content blocks (the problematic case)
    result1 = {
        'messages': [
            type('AIMessage', (), {
                'content': [
                    {
                        'type': 'text',
                        'text': "The following booking has a payment status of success ('S') but a booking status of failed ('F'):\n\n| Booking ID | Order Status | Payment Status |\n| :--- | :--- | :--- |\n| TJ5162540494 | F | S |",
                        'extras': {'signature': 'test123'}
                    }
                ]
            })()
        ]
    }
    
    content1 = extract_content_from_result(result1)
    print("Test 1 - Complex content blocks:")
    print(content1)
    print("\n" + "="*80 + "\n")
    
    # Test case 2: Simple string content
    result2 = {
        'messages': [
            type('AIMessage', (), {
                'content': "Simple text response"
            })()
        ]
    }
    
    content2 = extract_content_from_result(result2)
    print("Test 2 - Simple string:")
    print(content2)
    print("\n" + "="*80 + "\n")
    
    # Test case 3: Dict content with text key
    result3 = {
        'messages': [
            type('AIMessage', (), {
                'content': {
                    'text': "Text from dict",
                    'other': 'data'
                }
            })()
        ]
    }
    
    content3 = extract_content_from_result(result3)
    print("Test 3 - Dict content:")
    print(content3)
    print("\n" + "="*80 + "\n")
    
    print("✅ All extraction tests passed!")


if __name__ == "__main__":
    test_extract_content()
