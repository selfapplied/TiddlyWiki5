#!/usr/bin/env python3
"""
Export Demo for Symbolic Attractor Engine
========================================

Demonstrates different export formats for attractor analysis results.
"""

from attractor_explorer import AttractorExplorer
import json
import datetime

def export_as_toml(results, filename="results.toml"):
    """Export results in TOML format"""
    with open(filename, 'w') as f:
        f.write("# Symbolic Attractor Engine Results\n")
        f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n\n")
        
        for section_name, section_data in results.items():
            f.write(f"[{section_name}]\n")
            
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, list):
                        f.write(f"{key} = [\n")
                        for item in value:
                            if isinstance(item, list):
                                f.write(f"  [\"{item[0]}\", \"{item[1]}\"],\n")
                            else:
                                f.write(f"  \"{item}\",\n")
                        f.write("]\n")
                    else:
                        f.write(f"{key} = \"{value}\"\n")
            else:
                f.write(f"value = \"{section_data}\"\n")
            f.write("\n")
    
    print(f"✅ Exported to {filename}")

def export_as_json(results, filename="results.json"):
    """Export results in JSON format"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Exported to {filename}")

def export_as_text(results, filename="results.txt"):
    """Export results in human-readable text format"""
    with open(filename, 'w') as f:
        f.write("SYMBOLIC ATTRACTOR ENGINE RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
        
        for section_name, section_data in results.items():
            f.write(f"{section_name.upper()}\n")
            f.write("-" * len(section_name) + "\n")
            
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"{section_data}\n")
            f.write("\n")
    
    print(f"✅ Exported to {filename}")

def main():
    """Run the export demonstration"""
    print("🚀 Symbolic Attractor Engine - Export Demo")
    print("=" * 50)
    
    # Create explorer and run analysis
    explorer = AttractorExplorer()
    results = explorer.run_comprehensive_analysis()
    
    # Export in different formats
    print("\n📤 Exporting results in multiple formats...")
    
    # Convert results to exportable format
    exportable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            exportable_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    exportable_results[key][subkey] = subvalue
                else:
                    exportable_results[key][subkey] = str(subvalue)
        else:
            exportable_results[key] = str(value)
    
    # Export in different formats
    export_as_toml(exportable_results, "attractor_results.toml")
    export_as_json(exportable_results, "attractor_results.json")
    export_as_text(exportable_results, "attractor_results.txt")
    
    print("\n📊 Summary of exports:")
    print("   • TOML: attractor_results.toml (human-readable config format)")
    print("   • JSON: attractor_results.json (machine-readable format)")
    print("   • TEXT: attractor_results.txt (simple text format)")
    
    print("\n🎯 Analysis complete! Check all export files for results.")

if __name__ == "__main__":
    main() 