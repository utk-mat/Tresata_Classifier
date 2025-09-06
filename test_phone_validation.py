#!/usr/bin/env python3
"""
Test script to demonstrate enhanced phone number validation.
"""

from classifier import classify_value

def test_phone_validation():
    """Test various phone number formats and invalid cases."""
    
    test_cases = [
        # Valid phone numbers
        ("+1-555-123-4567", "phone"),      # US with country code
        ("555-123-4567", "phone"),         # US without country code
        ("+44 20 7946 0958", "phone"),     # UK
        ("+91 98765 43210", "phone"),      # India
        ("+49 30 12345678", "phone"),      # Germany
        ("+61 2 1234 5678", "phone"),      # Australia
        
        # Invalid phone numbers (should be classified as "other")
        ("123", "other"),                  # Too short
        ("+1-555-123", "other"),           # Too short for US
        ("+44 123", "other"),              # Too short for UK
        ("+91 12345", "other"),            # Too short for India
        ("abc-def-ghij", "other"),         # Invalid characters
        ("555-123-4567-999", "other"),     # Too long
        ("+999-555-123-4567", "other"),    # Invalid country code
        ("555-123-4567-extension-123", "other"),  # Invalid extension format
        
        # Edge cases
        ("+1 (555) 123-4567 ext 123", "phone"),  # Valid with extension
        ("555.123.4567", "phone"),         # Different separators
        ("(555) 123-4567", "phone"),       # Parentheses format
    ]
    
    print("🧪 Testing Enhanced Phone Number Validation")
    print("=" * 60)
    
    for value, expected in test_cases:
        result = classify_value(value)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{status} | Input: '{value}' | Expected: {expected} | Got: {result}")
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    test_phone_validation()
