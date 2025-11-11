# CMPE 257 Project

## Getting Started

This repository contains the raw data but a script must be run to 
obtain the processed data. From the root director run:

```bash
chmod +x scripts/preprocess_data.py
./scripts/preprocess_data.py
```

The first command ensures that the script is executable and the second command runs 
the script.

This script will process the raw stock data by grouping the data by company then
organizing the data into sliding windows of that company's stock. Each window is
accompanied by a target which represents the closing price of the stock on the day 
after the final observation in the window. Each processed data file represents a 
different size sliding window. 
