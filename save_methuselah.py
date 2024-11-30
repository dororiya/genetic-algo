import yaml
from yaml import SafeDumper
from yaml.loader import SafeLoader


# class to that save methuselahs in the yamal file he got as input
class SaveMethuselah(object):
    def __init__(self, file):
        if file.split('.')[-1] == 'yaml':
            self.config = load_config(file)
            self.file_path = file
        else:
            print('Not a yaml file')

    # Save the configuration
    def save_config(self):
        with open(self.file_path, 'w') as file:
            yaml.dump(self.config, file, Dumper=SafeDumper)

    # check if the list is in my yaml file
    def check_lst_exist(self, lst_to_check):
        for met_id in range(self.count_methuselah()):
            for lst in self.config['configuration']['methuselah'][met_id]['list_of_cell']:
                if lst_to_check == lst:
                    return True
        return False

    # Add new user to the configuration
    def add_methuselah_to_config(self, lst, time_to_converge, max_size):
        if 'configuration' not in self.config:
            self.config['configuration'] = {'methuselah': {}}
        if 'methuselah' not in self.config['configuration']:
            self.config['configuration']['methuselah'] = {}
            # Check if the list already exists
        for existing_id, methuselah in self.config['configuration']['methuselah'].items():
            if methuselah['list_of_cell'] == lst:
                return

            # Add a new Methuselah if the list is unique
        self.config['configuration']['methuselah'][self.count_methuselah()] = {
            'list_of_cell': lst,
            'time_to_converge': time_to_converge,
            'max_size': max_size,
        }
        self.save_config()

    # Count the number of Methuselahs in the configuration
    def count_methuselah(self):
        try:
            return len(self.config['configuration']['methuselah'])
        except KeyError:
            # Return 0 if the configuration or methuselah key doesn't exist
            return 0

    # Get a specific methuselah by ID
    def get_methuselah(self, methuselah_id):
        try:
            required_methuselah = self.config['configuration']['methuselah'][methuselah_id]
            # Convert each inner list to a tuple (maintains consistency)
            list_of_tuples = [tuple(inner_list) for inner_list in required_methuselah['list_of_cell']]
            return (list_of_tuples, required_methuselah['max_size'],
                    required_methuselah['time_to_converge'])
        except KeyError:
            print(f"Methuselah ID {methuselah_id} not found.")
            return None


# Load the configuration file
def load_config(file_path):
    try:
        with open(file_path) as file:
            my_config = yaml.load(file, Loader=SafeLoader)
            if my_config is None:  # Handle empty file case
                my_config = {}
            return my_config
    except FileNotFoundError:
        # Handle missing file case
        return {}



