from src.H5analyzer import H5Analyzer

def main():
    # Create an analyzer with the default file path
    analyzer = H5Analyzer()

    # Use the inspect() method
    print("Inspecting HDF5 file...")
    analyzer.inspect()

    # Use the play_video() method
    print("\nPlaying video from HDF5 file...")
    analyzer.play_video()


if __name__ == "__main__":
    main()