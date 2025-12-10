#!/usr/bin/env python3
"""
Test script to verify PDF parsing logic
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the parsing function
import importlib.util
spec = importlib.util.spec_from_file_location("vtu_automated_results", "vtu-automated-results.py")
vtu_automated_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vtu_automated_results)

def test_parsing():
    """Test the PDF parsing logic"""
    print("Testing PDF parsing logic...")
    
    # Get the project root directory
    project_root = Path(__file__).parent
    pdf_results_dir = project_root / "pdf_results"
    
    print(f"PDF Results Directory: {pdf_results_dir}")
    
    # Process all PDF files
    results = vtu_automated_results.process_pdf_batch(pdf_results_dir)
    
    print(f"Processed {len(results)} PDF files")
    
    # Display details of the first result
    if results:
        first_result = results[0]
        print("\nFirst result details:")
        print(f"USN: {first_result.get('usn', 'N/A')}")
        print(f"Student Name: {first_result.get('student_name', 'N/A')}")
        print(f"Semester: {first_result.get('semester', 'N/A')}")
        print(f"SGPA: {first_result.get('sgpa', 'N/A')}")
        print(f"CGPA: {first_result.get('cgpa', 'N/A')}")
        print(f"Subjects count: {len(first_result.get('subjects', []))}")
        
        # Display subject details
        for i, subject in enumerate(first_result.get('subjects', [])):
            print(f"  Subject {i+1}:")
            print(f"    Code: {subject.get('subject_code', 'N/A')}")
            print(f"    Name: {subject.get('subject_name', 'N/A')}")
            print(f"    Internal: {subject.get('internal', 'N/A')}")
            print(f"    External: {subject.get('external', 'N/A')}")
            print(f"    Total: {subject.get('total', 'N/A')}")
            print(f"    Result: {subject.get('result', 'N/A')}")

if __name__ == "__main__":
    test_parsing()