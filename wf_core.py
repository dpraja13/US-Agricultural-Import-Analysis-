import wf_dataprocessing
import wf_visualization

def main():
    print("Running data processing...")
    wf_dataprocessing.main()
    
    print("Running visualization...")
    wf_visualization.main()
    
    print("Data processing and visualization completed.")

if __name__ == "__main__":
    main()

