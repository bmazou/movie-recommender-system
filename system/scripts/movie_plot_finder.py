import pandas as pd


class MoviePlotFinder:
    def __init__(self, file_path="data/movie-plots-transformed.csv"):
        self.plots = self.load_plots(file_path)
        
    def load_plots(self, file_path):
        """Loads the movie plots from the given file path."""
        plots = pd.read_csv(file_path)
        plots.set_index('item_id', inplace=True)
        return plots

    def find_plot(self, item_id, restructure_output=True):
        """Finds the plot of the movie with the given item_id."""
        if item_id not in self.plots.index:
            print(f"Error: item_id {item_id} not found")
            return None
        
        plot = self.plots.loc[item_id, 'plot']
        title = self.plots.loc[item_id, 'title']
        
        return f'Plot of "{title}": {plot}' if restructure_output else plot