# Measles Data Visualizations

Five interactive charts displaying US measles surveillance data and vaccination coverage.

## Data Updates

**Current data** (CDC APIs): Updates automatically when available  
**Backup data**: Static JSON/CSV files used when APIs are unavailable  
**Historical data**: Static CSV files, manually updated

## Visualizations

### 1. Historical Timeline (1960-2025)
Annual confirmed measles cases with 9 annotated historical events. Square-root scale transforms case counts for visualization. Data points include vaccine licensure (1963), MMR introduction (1971), elimination goal (1978), two-dose recommendation (1989), elimination achieved (2000), Arizona outbreaks (2008, 2016), nationwide outbreak (2019), and first pediatric death in 22 years (2025).

### 2. Recent Trends (2015-2025) 
Bar chart of annual confirmed cases with overlaid line chart of MMR vaccination coverage percentages. Horizontal reference line at 95% marks herd immunity threshold. Dual y-axes show case counts (left) and coverage rates (right).

### 3. Disease Contagiousness Comparison
Dot plot displaying R₀ values: Ebola (2.0), HIV (4.0), COVID-19 Omicron (9.5), Chickenpox (12.0), Mumps (14.0), Measles (18.0). Each disease shows 20 dots in a circle with infected dots highlighted in proportion to R₀.

### 4. US State Map (2025)
Choropleth map colored by state MMR coverage rates with bubble overlay sized by confirmed case counts. States without coverage data appear gray. Bubble labels show case numbers, with "k" suffix for thousands.

### 5. Lives Saved Analysis (1975-2024)
Annual estimates of deaths prevented by measles vaccination using WHO EPI50 mathematical models. Theoretical projections comparing vaccine vs no-vaccine scenarios, not observed mortality data.

