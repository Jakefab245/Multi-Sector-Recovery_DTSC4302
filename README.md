# Multi-Sector Recovery: Economic Resiliency After Natural Disasters

## Problem Statement
There is no definitive metric measuring state-by-state recovery after Natural Disasters. Understanding how quickly states recover economically and socially after natural disasters is crucial for policy makers and disaster management agencies.

## Goal
To explore the factors that contribute to economic resiliency and create actionable insights on policy changes that will allow high-risk states to be able to recover quicker from natural disasters.

## Project Overview
This research project analyzes multi-sector recovery patterns following natural disasters across the United States. By examining economic, social, and political factors, we aim to identify key drivers of resilience and provide evidence-based recommendations for improving disaster recovery outcomes.

## Project Structure
- **/data** - Contains all datasets including disaster data, economic indicators, unemployment statistics, and political data
- **/visual_analysis** - Visualization and analysis code for graphs and results
- **generate_map.py** - FEMA Map generation script
- **main.ipynb** - Main project notebook with analysis
- **fema_map.html** - Interactive FEMA disaster map

## Key Datasets
- `disaster_data_export.csv` - Combined dataset for modeling
- `1976-2020-president.csv` - Political control data
- `HazardMitigationPlanStatuses.csv` - FEMA mitigation planning data
- Economic data from BEA (State GDP 1997-2024)
- Employment data from BLS (LAUS Unemployment 2006-2025)
- Disaster data from FEMA (2006-2025)
- Storm events data from NOAA (2005-2023)

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries: pandas, matplotlib, seaborn, folium, numpy, scikit-learn

### Installation
```bash
git clone https://github.com/Jakefab245/Multi-Sector-Recovery_DTSC4302.git
cd Multi-Sector-Recovery_DTSC4302
pip install -r requirements.txt
```

### Running the Analysis
1. Ensure all data files are in the `/data` directory
2. Open and run `main.ipynb` in Jupyter Notebook
3. View generated visualizations in `/visual_analysis`

## Contributing
This is a collaborative project. For issues or suggestions, please create an issue on GitHub.

## Contributors
- Ana Abreu (estvjana)
- Jake Fabrizio (Jakefab245)

## License
This project is part of DTSC4301-4302 coursework.
