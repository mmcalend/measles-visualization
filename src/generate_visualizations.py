import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math
import os
from datetime import datetime
import pytz
import requests
from pathlib import Path
import json

# =============================================================================
# ASU VISUAL IDENTITY AND STYLING CONFIGURATION
# =============================================================================

class ASUConfig:
    """
    Centralized configuration class for consistent ASU visual identity across all visualizations.
    
    Epidemiologists can modify colors, fonts, and layout parameters here to match 
    their institution's branding or publication requirements.
    """

    # Color Palette - Arizona State University Brand Colors
    PRIMARY_MAROON = "#8C1D40"     # ASU official maroon
    PRIMARY_GOLD = "#FFC627"       # ASU official gold
    BLACK = "#000000"
    WHITE = "#FFFFFF"
    DARK_GRAY = "#2C2C2C"         # For primary text and titles
    MEDIUM_GRAY = "#666666"        # For secondary text
    LIGHT_GRAY = "#CCCCCC"        # For borders and subtle elements
    BACKGROUND_GRAY = "#F8F8F8"   # Chart background
    SUCCESS = "#28A745"           # For positive indicators (e.g., herd immunity threshold)
    INFO = "#17A2B8"              # For informational elements

    # Typography Configuration
    FONT_FAMILY = "Arial, sans-serif"    # Web-safe font for broad compatibility
    MAIN_TITLE_SIZE = 22                 # Primary chart titles
    SUBTITLE_SIZE = 14                   # Chart subtitles and descriptions
    AXIS_TITLE_SIZE = 13                 # X and Y axis labels
    AXIS_TICK_SIZE = 12                  # Axis tick labels
    ANNOTATION_SIZE = 11                 # Data point annotations
    LEGEND_SIZE = 12                     # Legend text
    SOURCE_SIZE = 10                     # Data source citations
    FOOTER_COLOR = MEDIUM_GRAY           # Color for footnotes and sources

    # Layout and Spacing Configuration
    MARGIN_LEFT = 60          # Left margin for Y-axis labels
    MARGIN_RIGHT = 60         # Right margin for secondary Y-axis if needed
    MARGIN_TOP = 100          # Top margin for titles and subtitles
    MARGIN_BOTTOM = 180       # Bottom margin for data sources and notes
    GRID_COLOR = "rgba(0,0,0,0.1)"  # Subtle grid lines

    # Interactive Configuration for Web Embedding
    IFRAME_CONFIG = {
        "displayModeBar": False,    # Hide Plotly toolbar for cleaner embedding
        "responsive": True,         # Responsive design for different screen sizes
        "displaylogo": False,       # Remove Plotly logo
        "doubleClick": "reset",     # Double-click to reset zoom
        "scrollZoom": False         # Disable scroll zooming
    }

def create_base_figure(title="", subtitle=""):
    """
    Create a standardized base figure with consistent ASU styling.
    
    This function establishes the foundation for all visualizations, ensuring
    consistent typography, colors, and layout across different chart types.
    
    Parameters:
    -----------
    title : str
        Primary chart title (will be styled as main title)
    subtitle : str  
        Secondary descriptive text (will be styled as subtitle)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Base figure with ASU styling applied
    """

    # Build hierarchical title structure with HTML formatting
    title_text = f"<b style='color:{ASUConfig.DARK_GRAY}; font-size:{ASUConfig.MAIN_TITLE_SIZE}px'>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:{ASUConfig.SUBTITLE_SIZE}px; color:{ASUConfig.MEDIUM_GRAY}'>{subtitle}</span>"

    fig = go.Figure()

    # Apply comprehensive layout styling
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,                    # Center title horizontally
            xanchor="center",
            font=dict(family=ASUConfig.FONT_FAMILY)
        ),
        plot_bgcolor=ASUConfig.BACKGROUND_GRAY,    # Chart area background
        paper_bgcolor=ASUConfig.WHITE,             # Overall figure background
        font=dict(
            family=ASUConfig.FONT_FAMILY,
            size=ASUConfig.AXIS_TICK_SIZE,
            color=ASUConfig.DARK_GRAY
        ),
        margin=dict(
            l=ASUConfig.MARGIN_LEFT,
            r=ASUConfig.MARGIN_RIGHT,
            t=ASUConfig.MARGIN_TOP,
            b=ASUConfig.MARGIN_BOTTOM
        ),
        # Hover label styling for data point interactions
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=ASUConfig.PRIMARY_MAROON,
            font=dict(
                family=ASUConfig.FONT_FAMILY,
                size=ASUConfig.AXIS_TICK_SIZE
            )
        ),
        # Legend styling
        legend=dict(
            font=dict(
                family=ASUConfig.FONT_FAMILY,
                size=ASUConfig.LEGEND_SIZE,
                color=ASUConfig.DARK_GRAY
            ),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=ASUConfig.LIGHT_GRAY,
            borderwidth=1
        ),
        autosize=True,
        height=500                    # Standard height for consistency
    )

    return fig

def style_axes(fig, x_title="", y_title=""):
    """
    Apply consistent axis styling including titles, grids, and tick formatting.
    
    This function standardizes the appearance of axes across all chart types,
    ensuring readability and professional presentation.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure object to apply axis styling to
    x_title : str
        X-axis title text
    y_title : str
        Y-axis title text
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure with styled axes
    """

    # X-axis styling
    fig.update_xaxes(
        title=dict(
            text=f"<b>{x_title}</b>" if x_title else "",
            font=dict(
                size=ASUConfig.AXIS_TITLE_SIZE,
                color=ASUConfig.DARK_GRAY,
                family=ASUConfig.FONT_FAMILY
            )
        ),
        tickfont=dict(size=ASUConfig.AXIS_TICK_SIZE, color=ASUConfig.DARK_GRAY),
        showgrid=True,
        gridcolor=ASUConfig.GRID_COLOR,
        gridwidth=1,
        linecolor=ASUConfig.LIGHT_GRAY,
        showline=True
    )

    # Y-axis styling
    fig.update_yaxes(
        title=dict(
            text=f"<b>{y_title}</b>" if y_title else "",
            font=dict(
                size=ASUConfig.AXIS_TITLE_SIZE,
                color=ASUConfig.DARK_GRAY,
                family=ASUConfig.FONT_FAMILY
            )
        ),
        tickfont=dict(size=ASUConfig.AXIS_TICK_SIZE, color=ASUConfig.DARK_GRAY),
        showgrid=True,
        gridcolor=ASUConfig.GRID_COLOR,
        gridwidth=1,
        linecolor=ASUConfig.LIGHT_GRAY,
        showline=True
    )

    return fig

def add_comprehensive_data_source(fig, sources_dict, additional_text=""):
    """
    Add comprehensive data source citations with timestamp to visualizations.
    
    This function ensures proper attribution of data sources and provides
    transparency about when data was last updated, which is critical for
    epidemiological visualizations.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to add source annotation to
    sources_dict : dict or str
        Dictionary mapping source labels to citations, or single source string
    additional_text : str
        Additional context or notes to include
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure with source annotation added
    """
    
    # Get actual data refresh timestamp from backup logs
    refresh_time = get_data_refresh_time()
    
    source_text = "<b>Data Sources:</b><br>"

    # Handle different source input formats
    if isinstance(sources_dict, dict):
        for label, source in sources_dict.items():
            source_text += f"{label}: {source}<br>"
    else:
        if isinstance(sources_dict, str):
            source_text += sources_dict
        else:
            source_text += "<i>Invalid data source format provided.</i>"

    # Add supplementary context if provided
    if additional_text:
        source_text += f"<br>{additional_text}"
    
    # Add data freshness timestamp
    source_text += f"<br><br><i>Data last refreshed: {refresh_time}</i>"

    # Position source annotation at bottom-left of chart
    fig.add_annotation(
        text=source_text,
        xref="paper", yref="paper",
        x=0.0, y=-0.25,               # Position relative to chart area
        xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(
            family=ASUConfig.FONT_FAMILY,
            size=ASUConfig.SOURCE_SIZE,
            color=ASUConfig.FOOTER_COLOR
        ),
        align="left",
        # Styled background box for readability
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=ASUConfig.LIGHT_GRAY,
        borderwidth=1,
        borderpad=4
    )
    return fig

def get_data_refresh_time():
    """
    Retrieve the actual data refresh timestamp from backup logs.
    
    This function provides transparency about data freshness by checking
    when the most recent data backup was created.
    
    Returns:
    --------
    str
        Formatted timestamp of last data refresh in Arizona timezone
    """
    try:
        backup_log_path = Path('data/backups/backup_log.json')
        if backup_log_path.exists():
            with open(backup_log_path, 'r') as f:
                logs = json.load(f)
                if logs:
                    # Extract most recent backup timestamp
                    latest_log = logs[-1]
                    backup_timestamp = datetime.fromisoformat(latest_log['timestamp'].replace('Z', '+00:00'))
                    
                    # Convert to Arizona time (no daylight saving)
                    arizona_tz = pytz.timezone('US/Arizona')
                    arizona_time = backup_timestamp.astimezone(arizona_tz)
                    return arizona_time.strftime("%B %d, %Y at %I:%M %p MST")
    except Exception as e:
        print(f"Warning: Could not read backup log: {e}")
    
    # Fallback to current Arizona time if backup log unavailable
    arizona_tz = pytz.timezone('US/Arizona')
    arizona_time = datetime.now(arizona_tz)
    return arizona_time.strftime("%B %d, %Y at %I:%M %p MST")

def save_figure(fig, filename, output_dir="docs"):
    """
    Save visualization as HTML file optimized for web embedding.
    
    This function outputs publication-ready HTML files that can be embedded
    in websites, reports, or presentations.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to save
    filename : str
        Output filename (without extension)
    output_dir : str
        Output directory path
    """
    try:
        Path(output_dir).mkdir(exist_ok=True)
        filepath = Path(output_dir) / f"{filename}.html"
        
        # Save with CDN-hosted Plotly for smaller file sizes
        fig.write_html(
            str(filepath),
            include_plotlyjs='cdn',           # Use CDN instead of inline JS
            config=ASUConfig.IFRAME_CONFIG,   # Apply interactive configuration
            div_id=filename.replace('_', '-'),
            full_html=True,
            default_width='100%',
            default_height='500px'
        )
        print(f"✓ Saved: {filepath}")
        
        # Report file size for monitoring
        if filepath.exists():
            file_size = filepath.stat().st_size
            print(f"  File size: {file_size} bytes")
        else:
            print(f"ERROR: File was not created: {filepath}")
            
    except Exception as e:
        print(f"ERROR saving {filename}: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# DATA LOADING AND BACKUP MANAGEMENT
# =============================================================================

def load_backup_data(source_name, backup_dir="data/backups"):
    """
    Load data from backup files when primary sources are unavailable.
    
    This function implements resilient data loading by falling back to
    local backup files when live data sources fail, ensuring visualizations
    can still be generated during data source outages.
    
    Parameters:
    -----------
    source_name : str
        Name of data source (used in backup filenames)
    backup_dir : str
        Directory containing backup files
        
    Returns:
    --------
    pd.DataFrame
        Loaded data or empty DataFrame if no backup found
    """
    
    latest_file_json = Path(backup_dir) / f"{source_name}_latest.json"
    latest_file_csv = Path(backup_dir) / f"{source_name}_latest.csv"
    
    # Try JSON format first (preserves data types), then CSV
    for latest_file in [latest_file_json, latest_file_csv]:
        if latest_file.exists():
            try:
                if latest_file.suffix == '.json':
                    df = pd.read_json(str(latest_file))
                    print(f"✓ Loaded {source_name} from JSON backup: {len(df)} rows")
                    return df
                else:
                    df = pd.read_csv(str(latest_file))
                    print(f"✓ Loaded {source_name} from CSV backup: {len(df)} rows")
                    return df
            except Exception as e:
                print(f"Error loading backup for {source_name}: {e}")
    
    print(f"WARNING: No backup found for {source_name}")
    return pd.DataFrame()

def safe_load_csv(filepath, required_columns=None):
    """
    Safely load CSV files with comprehensive error handling and validation.
    
    This function provides robust CSV loading with column validation,
    which is essential for maintaining data integrity in epidemiological
    workflows where column names may vary between data sources.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    required_columns : list, optional
        List of column names that must be present
        
    Returns:
    --------
    pd.DataFrame
        Loaded data or empty DataFrame if loading fails
    """
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            
            # Debug: Display available columns for troubleshooting
            print(f"  Columns in {filepath}: {list(df.columns)}")
            
            # Validate required columns if specified
            if required_columns:
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"WARNING: {filepath} missing required columns: {missing_cols}")
                    return pd.DataFrame()
            
            print(f"✓ Loaded {filepath}: {len(df)} rows")
            return df
        else:
            print(f"WARNING: {filepath} not found")
            return pd.DataFrame()
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return pd.DataFrame()

def load_data():
    """
    Comprehensive data loading function with backup fallback support.
    
    This function loads all required datasets for measles visualizations,
    implementing a tiered approach: primary sources first, then backups.
    Epidemiologists can modify this function to add new data sources
    or change loading priorities.
    
    Returns:
    --------
    dict
        Dictionary containing all loaded datasets
    """
    try:
        print("Loading data files with backup fallback...")
        
        # Ensure data directory structure exists
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Initialize data storage dictionary
        data = {}
        
        # Load local static data files with validation
        local_files = {
            'timeline': ('data/timeline.csv', ['Year', 'Cases']),
            'mmr': ('data/MMRKCoverage.csv', ['year', 'Location']),
            'mmr_map': ('data/MMRKCoverage25.csv', ['Geography'])  # Note: Column will be renamed after loading
        }
        
        for key, (filepath, required_cols) in local_files.items():
            df = safe_load_csv(filepath, required_cols)
            # Handle column name standardization for map data
            if key == 'mmr_map' and not df.empty:
                df = df.rename(columns={'Geography': 'geography'})  
                print(f"  Renamed 'Geography' to 'geography' column")
            data[key] = df

        # Load live CDC data with backup fallback
        print("Loading CDC data with backup fallback...")
        
        # US Measles Cases - Primary endpoint
        try:
            print("Attempting to load fresh CDC measles cases data...")
            response = requests.get('https://www.cdc.gov/wcms/vizdata/measles/MeaslesCasesYear.json', timeout=30)
            response.raise_for_status()
            data['usmeasles'] = pd.read_json(response.text)
            print(f"✓ Loaded fresh CDC measles cases: {len(data['usmeasles'])} rows")
        except Exception as e:
            print(f"WARNING: Could not load fresh CDC measles data: {e}")
            data['usmeasles'] = load_backup_data('cdc_measles_cases')

        # US State-level Map Data - Primary endpoint with MMR coverage merge
        print("Loading US Map data...")
        try:
            response = requests.get('https://www.cdc.gov/wcms/vizdata/measles/MeaslesCasesMap.json', timeout=30)
            response.raise_for_status()
            usmap_cases = pd.read_json(response.text)
            
            # Merge with vaccination coverage data if available
            if not data['mmr_map'].empty:
                print(f"Merging map data: cases={len(usmap_cases)} rows, mmr_map={len(data['mmr_map'])} rows")
                usmap = usmap_cases.merge(data['mmr_map'], on='geography', how='left')
                print(f"After merge: {len(usmap)} rows")
                
                # Filter for current year data (2025 as of this implementation)
                usmap = usmap[usmap['year_x'] == 2025].copy()
                print(f"After 2025 filter: {len(usmap)} rows")
                
                # Convert vaccination data to numeric for analysis
                usmap['Estimate (%)'] = pd.to_numeric(usmap['Estimate (%)'], errors='coerce')
                data['usmap'] = usmap
                print(f"✓ Loaded US map data with vaccination info: {len(usmap)} rows")
                print(f"  Sample vaccination data: {usmap['Estimate (%)'].describe()}")
            else:
                print("WARNING: No MMR map data available for merge")
                data['usmap'] = usmap_cases
                
        except Exception as e:
            print(f"WARNING: Could not load fresh US map data: {e}")
            data['usmap'] = load_backup_data('cdc_measles_map')

        # Load WHO EPI50 Vaccine Impact Data - Mathematical modeling estimates
        print("Loading WHO vaccine impact data with backup fallback...")
        vaccine_files = {}
        
        # Load vaccine scenario data
        try:
            print("Attempting to load fresh WHO vaccine data...")
            vaccine_url = "https://raw.githubusercontent.com/WorldHealthOrganization/epi50-vaccine-impact/refs/tags/v1.0/extern/raw/epi50_measles_vaccine.csv"
            vaccine_files['vaccine'] = pd.read_csv(vaccine_url)
            print(f"✓ Loaded fresh WHO vaccine data: {len(vaccine_files['vaccine'])} rows")
        except Exception as e:
            print(f"WARNING: Could not load fresh WHO vaccine data: {e}")
            vaccine_files['vaccine'] = load_backup_data('who_vaccine_impact')

        # Load no-vaccine counterfactual scenario data
        try:
            print("Attempting to load fresh WHO no-vaccine data...")
            no_vaccine_url = "https://raw.githubusercontent.com/WorldHealthOrganization/epi50-vaccine-impact/refs/tags/v1.0/extern/raw/epi50_measles_no_vaccine.csv"
            vaccine_files['no_vaccine'] = pd.read_csv(no_vaccine_url)
            print(f"✓ Loaded fresh WHO no-vaccine data: {len(vaccine_files['no_vaccine'])} rows")
        except Exception as e:
            print(f"WARNING: Could not load fresh WHO no-vaccine data: {e}")
            vaccine_files['no_vaccine'] = load_backup_data('who_no_vaccine')

        # Process vaccine impact analysis if both scenarios loaded successfully
        if not vaccine_files['vaccine'].empty and not vaccine_files['no_vaccine'].empty:
            try:
                # Filter to US data only
                vax_usa = vaccine_files['vaccine'][vaccine_files['vaccine']['iso'] == 'USA'].copy()
                no_vax_usa = vaccine_files['no_vaccine'][vaccine_files['no_vaccine']['iso'] == 'USA'].copy()
                
                # Merge scenarios and calculate lives saved estimates
                merged_vaccine = pd.merge(no_vax_usa, vax_usa, on='year', suffixes=('_no_vaccine', '_vaccine'))
                merged_vaccine['lives_saved'] = merged_vaccine['mean_deaths_no_vaccine'] - merged_vaccine['mean_deaths_vaccine']
                merged_vaccine['lives_saved_ub'] = merged_vaccine['ub_deaths_no_vaccine'] - merged_vaccine['lb_deaths_vaccine']
                merged_vaccine['lives_saved_lb'] = merged_vaccine['lb_deaths_no_vaccine'] - merged_vaccine['ub_deaths_vaccine']
                merged_vaccine = merged_vaccine.sort_values('year')
                data['vaccine_impact'] = merged_vaccine
                print(f"✓ Processed WHO vaccine impact data: {len(merged_vaccine)} rows")
            except Exception as e:
                print(f"WARNING: Could not process WHO vaccine data: {e}")
                data['vaccine_impact'] = pd.DataFrame()
        else:
            print("WARNING: Missing WHO vaccine data - some visualizations may be incomplete")
            data['vaccine_impact'] = pd.DataFrame()

        print("✓ Data loading completed")
        
        # Debug: Print final data summary for verification
        print("\n=== FINAL DATA SUMMARY ===")
        for key, df in data.items():
            if isinstance(df, pd.DataFrame):
                print(f"  {key}: {len(df)} rows, columns: {list(df.columns)}")
            else:
                print(f"  {key}: {type(df)}")
        print("=========================\n")
        
        return data
        
    except Exception as e:
        print(f"ERROR in load_data: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# EPIDEMIOLOGICAL VISUALIZATION FUNCTIONS
# =============================================================================

def create_measles_timeline(timeline_data):
    """
    Create an enhanced timeline visualization showing measles cases with historical context.
    
    This function generates a comprehensive timeline that contextualizes measles
    case trends with key historical events like vaccine introductions, policy
    changes, and notable outbreaks. The square-root transformation allows
    visualization of both historical peaks and recent low-level trends.
    
    Parameters:
    -----------
    timeline_data : pd.DataFrame
        DataFrame containing columns: Year, Cases, and optionally Highlight
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive timeline visualization
    """
    
    if timeline_data.empty:
        return create_base_figure("Timeline", "No timeline data available")

    # Prepare and clean data
    df = timeline_data.copy()

    def wrap_text(text, width=30):
        """
        Wrap annotation text to prevent overlapping and improve readability.
        
        Parameters:
        -----------
        text : str
            Text to wrap
        width : int
            Maximum characters per line
            
        Returns:
        --------
        str
            HTML-formatted text with line breaks
        """
        if pd.isna(text) or text == "":
            return None
        words = str(text).split()
        lines, line, line_len = [], [], 0
        for w in words:
            if line_len + len(w) + 1 <= width:
                line.append(w)
                line_len += len(w) + 1
            else:
                if line:
                    lines.append(" ".join(line))
                line = [w]
                line_len = len(w)
        if line:
            lines.append(" ".join(line))
        return "<br>".join(lines)

    # Validate required columns
    required_cols = ['Year', 'Cases']
    if not all(col in df.columns for col in required_cols):
        print(f"WARNING: Timeline missing required columns. Has: {list(df.columns)}")
        return create_base_figure("Timeline", "Required data columns missing")

    # Process highlight annotations if available
    if "Highlight" in df.columns:
        df["Label_wrapped"] = df["Highlight"].apply(wrap_text)
        has_highlights = df["Label_wrapped"].notna()
    else:
        df["Label_wrapped"] = None
        has_highlights = pd.Series([False] * len(df))

    # Apply square-root transformation to accommodate wide range of case counts
    df['Cases_sqrt'] = np.sqrt(df['Cases'])

    # Create descriptive subtitle
    subtitle_text = "Confirmed Measles Cases and Historical Highlights (1960-2025)<br>Event Types: ● National Events • ♦ Arizona-Specific Events"

    fig = create_base_figure(
        "A Timeline of Measles in the United States",
        subtitle_text
    )

    # Add subtle background gradient for visual appeal
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        fillcolor=f"rgba(140, 29, 64, 0.05)",
        layer="below", line_width=0,
    )

    # Add decade grid lines for temporal reference
    years = df["Year"].values
    y_max = df["Cases_sqrt"].max()
    for year in range(int(years.min()), int(years.max()) + 1, 10):
        fig.add_shape(
            type="line",
            x0=year, y0=0, x1=year, y1=y_max * 1.1,
            line=dict(color=ASUConfig.GRID_COLOR, width=1, dash="dot"),
            layer="below"
        )

    # Main trend line with spline smoothing
    fig.add_trace(go.Scatter(
        x=df["Year"],
        y=df["Cases_sqrt"],
        mode="lines",
        line=dict(color=ASUConfig.PRIMARY_MAROON, width=5, shape="spline", smoothing=0.4),
        hovertemplate="<b>%{x}</b><br><b>Measles Cases:</b> %{customdata:,}<extra></extra>",
        customdata=df['Cases'],
        name="Annual Measles Cases",
        showlegend=False
    ))

    highlight_data = df[has_highlights]

    if not highlight_data.empty:
        # Distinguish between national and Arizona-specific events
        arizona_years = [2008, 2016]  # Years with Arizona-specific outbreak events
        highlight_data = highlight_data.copy()
        highlight_data['is_arizona'] = highlight_data['Year'].isin(arizona_years)

        # Add markers for national historical events (circles)
        national_events = highlight_data[~highlight_data['is_arizona']]
        if not national_events.empty:
            # Outer ring for national events
            fig.add_trace(go.Scatter(
                x=national_events["Year"],
                y=national_events["Cases_sqrt"],
                mode="markers",
                marker=dict(
                    size=16, color=ASUConfig.WHITE,
                    line=dict(width=3, color=ASUConfig.PRIMARY_MAROON),
                    symbol="circle"
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "<b>Cases:</b> %{customdata:,}<br>"
                    "<b>Historical Event:</b><br>%{text}<br>"
                    "<extra></extra>"
                ),
                customdata=national_events['Cases'],
                text=national_events['Label_wrapped'],
                name="Historical Events",
                showlegend=False
            ))

            # Inner marker for national events
            fig.add_trace(go.Scatter(
                x=national_events["Year"],
                y=national_events["Cases_sqrt"],
                mode="markers",
                marker=dict(size=8, color=ASUConfig.PRIMARY_GOLD, symbol="circle"),
                hoverinfo='skip',
                showlegend=False
            ))

        # Add markers for Arizona-specific events (diamonds)
        arizona_events = highlight_data[highlight_data['is_arizona']]
        if not arizona_events.empty:
            # Outer ring for Arizona events
            fig.add_trace(go.Scatter(
                x=arizona_events["Year"],
                y=arizona_events["Cases_sqrt"],
                mode="markers",
                marker=dict(
                    size=16, color=ASUConfig.WHITE,
                    line=dict(width=3, color=ASUConfig.PRIMARY_MAROON),
                    symbol="diamond"
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "<b>Cases:</b> %{customdata:,}<br>"
                    "<b>Historical Event:</b><br>%{text}<br>"
                    "<extra></extra>"
                ),
                customdata=arizona_events['Cases'],
                text=arizona_events['Label_wrapped'],
                name="Historical Events",
                showlegend=False
            ))

            # Inner marker for Arizona events
            fig.add_trace(go.Scatter(
                x=arizona_events["Year"],
                y=arizona_events["Cases_sqrt"],
                mode="markers",
                marker=dict(size=8, color=ASUConfig.PRIMARY_GOLD, symbol="diamond"),
                hoverinfo='skip',
                showlegend=False
            ))

        # Add always-visible annotations with year and case count
        def format_number(num):
            """Format large numbers with appropriate suffixes for readability"""
            if num >= 1_000_000:
                return f"{num/1_000_000:.1f}M"
            elif num >= 1_000:
                return f"{num/1_000:.1f}K"
            else:
                return f"{num:,}"

        annotations = []
        highlight_years = highlight_data["Year"].values

        # Create staggered annotations to prevent overlap
        for idx, (_, row) in enumerate(highlight_data.iterrows()):
            # Alternate annotation positions to reduce visual clutter
            if len(highlight_years) > 6:
                y_offset = -50 if idx % 2 == 0 else -80
                x_offset = 15 if idx % 2 == 0 else -15
            else:
                y_offset = -60 if idx % 2 == 0 else -90
                x_offset = 0

            annotation_text = (
                f"<b>{int(row['Year'])}</b><br>"
                f"<span style='font-size:{ASUConfig.ANNOTATION_SIZE}px;'>{format_number(row['Cases'])} cases</span>"
            )

            annotations.append(dict(
                x=row["Year"],
                y=row["Cases_sqrt"],
                text=annotation_text,
                showarrow=True,
                arrowhead=1,
                arrowsize=0.5,
                arrowwidth=1,
                arrowcolor=ASUConfig.MEDIUM_GRAY,
                ax=x_offset,
                ay=y_offset,
                font=dict(family=ASUConfig.FONT_FAMILY, size=ASUConfig.ANNOTATION_SIZE, color=ASUConfig.PRIMARY_MAROON),
                align="center",
                opacity=0.9
            ))

        fig.update_layout(annotations=annotations)

    # Configure axes for timeline-specific requirements
    fig.update_yaxes(title="", showgrid=False, showticklabels=False, zeroline=False, showline=False)
    fig.update_xaxes(
        title=dict(
            text=f"<b style='color:{ASUConfig.DARK_GRAY}; font-size:{ASUConfig.AXIS_TITLE_SIZE}px;'>Year</b>",
            standoff=20
        ),
        showgrid=False,
        dtick=5,                              # Show every 5th year
        tickfont=dict(size=ASUConfig.AXIS_TICK_SIZE, color=ASUConfig.DARK_GRAY),
        tickmode='linear',
        showline=True,
        linewidth=2,
        linecolor=ASUConfig.LIGHT_GRAY,
        mirror=False
    )

    # Comprehensive data source attribution
    sources = {
        "Historical Data (1960-2024)": "Public Health Reports; US Census Bureau; CDC - processed by Our World in Data",
        "Current Data (2025)": '<a href="https://www.cdc.gov/measles/data-research/index.html" target="_blank">CDC Measles Surveillance</a>',
        "Historical References":
            '<a href="https://historyofvaccines.org/history/measles/timeline" target="_blank">(1) History of Vaccines</a> • '
            '<a href="https://www.cdc.gov/measles/about/history.html" target="_blank">(2) CDC History</a> • '
            '<a href="https://academic.oup.com/jid/article/203/11/1517/862546" target="_blank">(3) Journal of Infectious Diseases</a> • '
            '<a href="https://www.cdc.gov/mmwr/volumes/66/wr/mm6620a5.htm" target="_blank">(4) MMWR Weekly</a> • '
            '<a href="https://www.cdc.gov/mmwr/volumes/68/wr/mm6840e2.htm" target="_blank">(5) MMWR Weekly</a>'
    }

    note_text = "<i>Note: Chart uses square-root scale to show both historical peaks and recent trends</i>"

    fig = add_comprehensive_data_source(fig, sources, note_text)

    return fig

def create_recent_trends(usmeasles_data, mmr_data):
    """
    Create dual-axis visualization showing recent measles cases and vaccination coverage.
    
    This function generates a bar chart of annual measles cases overlaid with
    vaccination coverage trends, including the critical 95% herd immunity
    threshold. This visualization is essential for understanding the relationship
    between vaccination rates and disease incidence.
    
    Parameters:
    -----------
    usmeasles_data : pd.DataFrame
        DataFrame with columns: year, cases
    mmr_data : pd.DataFrame
        DataFrame with columns: year, Location, MMR (vaccination coverage)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Dual-axis bar and line chart
    """
    
    if usmeasles_data.empty and mmr_data.empty:
        fig = create_base_figure("Recent Trends", "No data available")
        return fig
    
    if usmeasles_data.empty:
        print("WARNING: No US measles data available")
        fig = create_base_figure("Recent Trends", "US measles data not available")
        return fig
    
    # Data preparation and cleaning
    usmeasles_clean = usmeasles_data.copy()
    usmeasles_clean['Location'] = 'United States'
    usmeasles_clean = usmeasles_clean.drop_duplicates(subset=['year'])
    
    # Merge with vaccination data if available
    if not mmr_data.empty:
        mmr_clean = mmr_data.copy().drop_duplicates(subset=['year', 'Location'])
        merged = pd.merge(usmeasles_clean, mmr_clean, on=['year', 'Location'], how='left')
    else:
        merged = usmeasles_clean.copy()
    
    # Filter to recent decade plus current year
    us_data = merged[merged['year'] > 2014].copy().drop_duplicates(subset=['year'])
    us_data = us_data.sort_values('year').reset_index(drop=True)

    if us_data.empty:
        fig = create_base_figure("Recent Trends", "No data available for recent years")
        return fig

    fig = create_base_figure(
        "Measles Cases and Vaccination Coverage in the United States",
        "2015-2025"
    )

    # Validate essential columns
    if 'cases' not in us_data.columns:
        print(f"Available columns: {list(us_data.columns)}")
        fig = create_base_figure("Recent Trends", "Cases data not available")
        return fig

    # Data type conversion and cleaning
    us_data['year'] = pd.to_numeric(us_data['year'], errors='coerce')
    us_data['cases'] = pd.to_numeric(us_data['cases'], errors='coerce')
    us_data = us_data.dropna(subset=['year', 'cases'])
    
    if us_data.empty:
        fig = create_base_figure("Recent Trends", "No valid data after cleaning")
        return fig

    # Create gradient-colored bars based on case count intensity
    max_cases = us_data["cases"].max() if us_data["cases"].max() > 0 else 1
    bar_colors = [
        f"rgba(140, 29, 64, {0.5 + 0.4 * (cases/max_cases)})"
        for cases in us_data["cases"]
    ]

    # Primary visualization: Cases as bars
    fig.add_trace(go.Bar(
        x=us_data["year"],
        y=us_data["cases"],
        name="Confirmed Measles Cases",
        marker=dict(color=bar_colors, line=dict(color=ASUConfig.PRIMARY_MAROON, width=1)),
        text=[f"{int(c):,}" if c > 100 else "" for c in us_data["cases"]],  # Show labels for significant outbreaks
        textposition="auto",
        textfont=dict(size=ASUConfig.ANNOTATION_SIZE, color=ASUConfig.WHITE),
        hovertemplate="<b>Year:</b> %{x}<br><b>Cases:</b> %{y:,}<extra></extra>"
    ))

    # Secondary visualization: Vaccination coverage line (if data available)
    has_mmr_data = False
    if 'MMR' in us_data.columns and not us_data['MMR'].isna().all():
        us_data['MMR'] = pd.to_numeric(us_data['MMR'], errors='coerce')
        valid_mmr_data = us_data.dropna(subset=['MMR'])
        
        if not valid_mmr_data.empty:
            has_mmr_data = True
            fig.add_trace(go.Scatter(
                x=valid_mmr_data["year"],
                y=valid_mmr_data["MMR"],
                name="MMR Coverage (%)",
                mode="lines+markers",
                line=dict(color=ASUConfig.PRIMARY_GOLD, width=3, shape="spline", smoothing=0.2),
                marker=dict(size=8, color=ASUConfig.PRIMARY_GOLD, line=dict(color=ASUConfig.BLACK, width=2)),
                hovertemplate="<b>Year:</b> %{x}<br><b>Coverage:</b> %{y:.1f}%<extra></extra>",
                yaxis="y2"                    # Plot on secondary Y-axis
            ))
            
            # Critical epidemiological reference: 95% herd immunity threshold
            fig.add_hline(
                y=95, 
                line=dict(dash="dash", color=ASUConfig.SUCCESS, width=2),
                opacity=0.7,
                yref="y2",
                annotation=dict(
                    text="<b>95% Herd Immunity Threshold</b>",
                    font=dict(color=ASUConfig.WHITE, size=ASUConfig.ANNOTATION_SIZE),
                    bgcolor=ASUConfig.SUCCESS,
                    bordercolor=ASUConfig.WHITE, 
                    borderwidth=1, 
                    borderpad=6
                ),
                annotation_position="top right"
            )

    # Apply standardized axis styling
    fig = style_axes(fig, "Year", "Confirmed Measles Cases")
    
    # Configure temporal X-axis
    fig.update_xaxes(
        dtick=2,                              # Show every other year
        showgrid=True,
        gridcolor=ASUConfig.GRID_COLOR,
        range=[us_data["year"].min() - 0.5, us_data["year"].max() + 0.5]
    )
    
    # Configure primary Y-axis for cases
    fig.update_yaxes(
        showgrid=True,
        gridcolor=ASUConfig.GRID_COLOR,
        range=[0, max(us_data["cases"]) * 1.1]
    )
    
    # Configure secondary Y-axis for vaccination coverage
    if has_mmr_data:
        fig.update_layout(
            yaxis2=dict(
                title=dict(
                    text="<b>MMR Vaccination Coverage (%)</b>",
                    font=dict(
                        color=ASUConfig.DARK_GRAY, 
                        size=ASUConfig.AXIS_TITLE_SIZE,
                        family=ASUConfig.FONT_FAMILY
                    )
                ),
                overlaying="y",
                side="right",
                range=[85, 100],              # Focus on epidemiologically relevant range
                showgrid=False,               # Avoid grid line conflicts
                tickfont=dict(color=ASUConfig.DARK_GRAY, size=ASUConfig.AXIS_TICK_SIZE),
                linecolor=ASUConfig.LIGHT_GRAY
            )
        )

    # Position legend for dual-axis readability
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.02, 
            y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=ASUConfig.LIGHT_GRAY,
            borderwidth=1,
            font=dict(
                family=ASUConfig.FONT_FAMILY, 
                size=ASUConfig.LEGEND_SIZE, 
                color=ASUConfig.DARK_GRAY
            )
        )
    )

    # Comprehensive source attribution
    sources = {
        "MMR Vaccination Coverage": '<a href="https://data.cdc.gov/Vaccinations/Vaccination-Coverage-and-Exemptions-among-Kinderga/ijqb-a7ye/about_data" target="_blank">CDC NIS</a>',
        "Measles Cases & Herd Immunity Reference": '<a href="https://www.cdc.gov/measles/data-research/index.html" target="_blank">CDC Surveillance</a>'
    }

    fig = add_comprehensive_data_source(fig, sources)

    return fig

def create_rnaught_comparison():
    """
    Create comparative visualization of basic reproduction numbers (R₀) across diseases.
    
    This function generates an intuitive dot plot showing how many people each
    infected person could potentially infect for different diseases. The visualization
    uses a "20 people" model where each circle represents 20 people, with the
    gold dot as the index case and maroon dots showing potential transmissions.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Dot plot comparing R₀ values across diseases
    """

    # Epidemiologically relevant R₀ values from peer-reviewed literature
    rknot = {
        'Disease': ['Ebola', 'HIV', 'COVID-19 (Omicron)', 'Chickenpox', 'Mumps', 'Measles'],
        'Rknot': [2, 4, 9.5, 12, 14, 18]
    }
    df = pd.DataFrame(rknot)

    fig = create_base_figure(
        "How Contagious are Different Diseases?",
        "Each circle shows 20 people. The gold dot is the first infected person. Maroon dots show potential infections (R₀)"
    )

    # Layout configuration for responsive design
    TOTAL_DISEASES = len(df)
    X_SPACING = 5                  # Horizontal spacing between disease groups
    Y_POSITION = 0                 # Vertical center line

    # Visual parameters for dot plot
    TOTAL_DOTS = 20               # Total people represented in each circle
    DOT_SIZE = 12                 # Individual person dot size
    CIRCLE_RADIUS = 1.3           # Radius of person arrangement
    CENTER_DOT_SIZE = 22          # Index case (central) dot size

    # Color scheme for transmission visualization
    INFECTED_COLOR = ASUConfig.PRIMARY_MAROON    # People who could be infected
    NOT_INFECTED_COLOR = ASUConfig.LIGHT_GRAY    # People who remain uninfected
    INDEX_CASE_COLOR = ASUConfig.PRIMARY_GOLD    # Original infected person

    # Generate visualization for each disease
    for i, (disease, r0) in enumerate(zip(df['Disease'], df['Rknot'])):
        cx = i * X_SPACING            # Center X coordinate for this disease
        cy = Y_POSITION               # Center Y coordinate

        # Calculate positions for 20 people in circular arrangement
        angles = np.linspace(0, 2 * math.pi, TOTAL_DOTS, endpoint=False)
        x_coords = cx + CIRCLE_RADIUS * np.cos(angles)
        y_coords = cy + CIRCLE_RADIUS * np.sin(angles)

        # Add dots representing individual people
        for j in range(TOTAL_DOTS):
            if j < r0:                # This person could be infected based on R₀
                dot_color = INFECTED_COLOR
                hover_text = "This person could be infected"
                hover_bgcolor = ASUConfig.PRIMARY_MAROON
                hover_font_color = ASUConfig.WHITE
            else:                     # This person remains uninfected
                dot_color = NOT_INFECTED_COLOR
                hover_text = "This person is not infected"
                hover_bgcolor = ASUConfig.LIGHT_GRAY
                hover_font_color = ASUConfig.DARK_GRAY

            fig.add_trace(go.Scatter(
                x=[x_coords[j]], y=[y_coords[j]],
                mode='markers',
                marker=dict(size=DOT_SIZE, color=dot_color, line=dict(width=1, color=ASUConfig.WHITE)),
                hoverinfo='text',
                text=hover_text,
                hoverlabel=dict(bgcolor=hover_bgcolor, font=dict(color=hover_font_color)),
                showlegend=False
            ))

        # Add central index case (patient zero)
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode='markers',
            marker=dict(size=CENTER_DOT_SIZE, color=INDEX_CASE_COLOR, line=dict(color=ASUConfig.WHITE, width=2)),
            hoverinfo='text',
            text=f"Original infected person<br><b>{disease}</b> (R₀ = {r0})",
            hoverlabel=dict(bgcolor=ASUConfig.PRIMARY_GOLD, font=dict(color=ASUConfig.BLACK)),
            showlegend=False
        ))

        # Add transmission lines from index case to potentially infected individuals
        line_x, line_y = [], []
        for j in range(TOTAL_DOTS):
            if j < r0:                # Draw connection line to potentially infected person
                line_x.extend([cx, x_coords[j], None])  # None creates line break
                line_y.extend([cy, y_coords[j], None])

        if line_x:
            fig.add_trace(go.Scatter(
                x=line_x, y=line_y,
                mode='lines',
                line=dict(width=1, color=INFECTED_COLOR),
                hoverinfo='skip',
                showlegend=False
            ))

        # Add disease label with R₀ value
        fig.add_annotation(
            x=cx, y=cy - CIRCLE_RADIUS - 1.0,
            text=f"<b>{disease}</b><br>R₀ = {r0}",
            showarrow=False,
            xanchor="center", yanchor="top",
            font=dict(size=ASUConfig.ANNOTATION_SIZE, color=ASUConfig.DARK_GRAY, family=ASUConfig.FONT_FAMILY),
            align="center"
        )

    # Calculate layout bounds for proper display
    x_min = -CIRCLE_RADIUS - 1.0
    x_max = (TOTAL_DISEASES - 1) * X_SPACING + CIRCLE_RADIUS + 1.0
    y_min = Y_POSITION - CIRCLE_RADIUS - 2.5
    y_max = Y_POSITION + CIRCLE_RADIUS + 1.0

    # Configure dot plot layout (hide axes, maintain aspect ratio)
    fig.update_layout(
        xaxis=dict(visible=False, range=[x_min, x_max]),
        yaxis=dict(visible=False, range=[y_min, y_max], scaleanchor="x", scaleratio=1),
        showlegend=False,
    )

    # Source attribution for R₀ values
    sources = {
        "University of Michigan School of Public Health": '<a href="https://sph.umich.edu/pursuit/2020posts/how-scientists-quantify-outbreaks.html" target="_blank">(Ebola, Measles)</a>',
        "Liu and Rocklöv, 2022": '<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8992231/" target="_blank">(COVID-19 Omicron Variant)</a>',
        "Journal of Theoretical Biology": '<a href="https://www.sciencedirect.com/science/article/abs/pii/S0022519399910640?via%3Dihub" target="_blank">(Chickenpox, Mumps)</a>',
        "Proceedings of the National Academy of Sciences": '<a href="https://www.pnas.org/content/pnas/111/45/16202.full.pdf" target="_blank">(HIV)</a>'
    }

    fig = add_comprehensive_data_source(fig, sources)

    return fig

def create_us_map(usmap_data):
    """
    Create choropleth map showing state-level measles cases and vaccination rates.
    
    This function generates a comprehensive map visualization combining:
    - Choropleth coloring by vaccination coverage rates
    - Proportional bubble markers for case counts
    - State-level public health data integration
    
    Parameters:
    -----------
    usmap_data : pd.DataFrame
        DataFrame containing geography, cases, and vaccination data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive US map with dual data encoding
    """
    
    if usmap_data.empty:
        return create_base_figure("US Map", "No map data available")

    # State name to FIPS code mapping for choropleth rendering
    state_codes = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
        'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
        'Wisconsin': 'WI', 'Wyoming': 'WY', 'New York City': 'NYC'
    }

    df = usmap_data.copy()
    
    # Validate essential geographic data
    if 'geography' not in df.columns:
        print("WARNING: 'geography' column not found in map data")
        print(f"Available columns: {list(df.columns)}")
        return create_base_figure("US Map", "Geography data not available")

    df["state_code"] = df["geography"].map(state_codes)

    # Identify cases data column (flexible column naming)
    cases_col = None
    for col in ['cases_calendar_year', 'cases', 'Cases']:
        if col in df.columns:
            cases_col = col
            print(f"Using cases column: {col}")
            break
    
    if cases_col is None:
        print("WARNING: No cases column found in map data")
        print(f"Available columns: {list(df.columns)}")
        return create_base_figure("US Map", "Cases data not available")

    fig = create_base_figure(
        "2025 Measles Cases and Vaccination Rates by State",
        "Circle size = Cases • Color = MMR Vaccination Rate"
    )

    # ASU-branded color scale for vaccination coverage
    asu_colorscale = [
        [0, ASUConfig.PRIMARY_MAROON],    # Low vaccination (concerning)
        [0.5, "#B8336A"],                # Intermediate
        [1, ASUConfig.PRIMARY_GOLD]      # High vaccination (ideal)
    ]

    # Identify vaccination coverage data column
    vaccination_col = None
    for col in ['Estimate (%)', 'vaccination_rate', 'mmr_coverage']:
        if col in df.columns:
            vaccination_col = col
            print(f"Using vaccination column: {col}")
            break

    # Create choropleth layer if vaccination data is available
    if vaccination_col is not None:
        print("Creating choropleth with vaccination data...")
        try:
            # Generate choropleth using Plotly Express for proper state mapping
            choropleth = px.choropleth(
                df, locations="state_code", locationmode="USA-states",
                color=vaccination_col, scope="usa",
                color_continuous_scale=asu_colorscale,
                labels={vaccination_col: "Vaccination Rate (%)"}
            )

            # Extract and configure choropleth trace
            choropleth_trace = choropleth.data[0]
            choropleth_trace.hovertemplate = f"<b>%{{customdata[2]}}</b><br>Cases: %{{customdata[0]:.0f}}<br>Vaccination Rate: %{{customdata[1]:.1f}}%<extra></extra>"
            choropleth_trace.customdata = df[[cases_col, vaccination_col, 'geography']]
            fig.add_trace(choropleth_trace)

            # Apply choropleth color axis configuration
            fig.update_layout(coloraxis=choropleth.layout.coloraxis)
            fig.update_layout(coloraxis_colorbar=dict(title="<b>Vaccination Rate (%)</b>"))
            print("✓ Choropleth created successfully")
            
        except Exception as e:
            print(f"ERROR creating choropleth: {e}")
    else:
        print("WARNING: No vaccination data found - map will show cases only")
        print(f"Available columns: {list(df.columns)}")

    # Configure map projection to focus on continental US
    fig.update_layout(geo=dict(scope="usa"))

    # Calculate bubble sizes based on case counts with epidemiological relevance
    max_cases = df[cases_col].max()
    min_cases = df[df[cases_col] > 0][cases_col].min() if any(df[cases_col] > 0) else 1

    def calculate_bubble_size(cases):
        """
        Calculate bubble size based on epidemiological significance.
        
        Bubble sizes are scaled to reflect public health impact:
        - Single cases: Small but visible (travel-related imports)
        - Small clusters (2-10): Medium size (limited transmission)
        - Outbreaks (>10): Large size (community transmission)
        """
        if cases <= 0:
            return 0
        elif cases == 1:
            return 10             # Single imported case
        elif cases <= 5:
            return 14             # Small cluster
        elif cases <= 10:
            return 20             # Moderate cluster
        elif cases <= 50:
            return 28             # Small outbreak
        elif cases <= 100:
            return 38             # Moderate outbreak
        elif cases <= 500:
            return 48             # Large outbreak
        else:                     # Major outbreak (500+)
            return 58

    # Apply bubble sizing calculation
    df["bubble_size"] = df[cases_col].apply(calculate_bubble_size)

    def format_label(cases):
        """Format case count labels for bubble display"""
        if cases <= 0:
            return ''
        elif cases < 1000:
            return str(int(cases))
        else:
            return f'{int(cases/1000)}k'

    df["text_label"] = df[cases_col].apply(format_label)

    # Calculate responsive font sizes for bubble labels
    def calculate_font_size(cases, bubble_size):
        """Scale font size proportionally to bubble size for readability"""
        if cases <= 0:
            return ASUConfig.ANNOTATION_SIZE
        elif cases == 1:
            return 8
        elif cases <= 5:
            return 9
        elif cases <= 10:
            return 10
        elif cases <= 50:
            return 11
        elif cases <= 100:
            return 12
        else:
            return 14

    df["font_size"] = df.apply(lambda row: calculate_font_size(row[cases_col], row["bubble_size"]), axis=1)

    # Configure bubble trace hover data based on available information
    if vaccination_col is not None:
        customdata = df[[cases_col, vaccination_col, 'geography']]
        hovertemplate = f"<b>%{{customdata[2]}}</b><br>Cases: %{{customdata[0]:.0f}}<br>Vaccination Rate: %{{customdata[1]:.1f}}%<extra></extra>"
    else:
        customdata = df[[cases_col, 'geography']]
        hovertemplate = f"<b>%{{customdata[1]}}</b><br>Cases: %{{customdata[0]:.0f}}<extra></extra>"

    # Add case count bubbles as scatter geo layer
    fig.add_trace(go.Scattergeo(
        locationmode="USA-states",
        locations=df["state_code"],
        text=df["text_label"],
        marker=dict(
            size=df["bubble_size"],
            color=ASUConfig.WHITE,           # White bubbles for contrast
            line=dict(width=2, color=ASUConfig.BLACK),  # Black borders for definition
            sizemode="diameter",
            opacity=0.8
        ),
        name="Cases",
        hovertemplate=hovertemplate,
        customdata=customdata,
        mode='markers+text',
        textposition="middle center",
        textfont=dict(
            size=df["font_size"],
            color=ASUConfig.BLACK,
            family=ASUConfig.FONT_FAMILY
        ),
        showlegend=True
    ))

    # Position legend for map readability
    fig.update_layout(
        legend=dict(x=0.02, y=0.3, bgcolor="rgba(255,255,255,0.9)",
                   bordercolor=ASUConfig.LIGHT_GRAY, borderwidth=1)
    )

    # Comprehensive data source attribution
    sources = {
        "MMR Vaccination Coverage": '<a href="https://data.cdc.gov/Vaccinations/Vaccination-Coverage-and-Exemptions-among-Kinderga/ijqb-a7ye/about_data" target="_blank">CDC NIS</a>',
        "Measles Cases": '<a href="https://www.cdc.gov/measles/data-research/index.html" target="_blank">CDC Surveillance</a>'
    }

    note_text = "<i>Note: Gray states did not have reported vaccination coverage for the 2024-2025 school year</i>"

    fig = add_comprehensive_data_source(fig, sources, note_text)

    return fig
    
def create_lives_saved_chart(vaccine_impact_data):
    """
    Create bar chart visualization of estimated lives saved by vaccination programs.
    
    This function displays WHO EPI50 mathematical modeling estimates showing
    the theoretical number of deaths that would have been prevented by measles
    vaccination programs compared to a counterfactual scenario with no vaccines.
    These are modeled estimates, not observed data.
    
    Parameters:
    -----------
    vaccine_impact_data : pd.DataFrame
        DataFrame containing vaccine impact modeling results with columns
        for year and lives_saved estimates
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart showing annual estimated lives saved
    """
    
    if vaccine_impact_data.empty:
        print("DEBUG - Lives saved: vaccine_impact_data is empty")
        return create_base_figure("Lives Saved", "No vaccine impact data available")

    df = vaccine_impact_data.copy()
    
    print(f"DEBUG - Lives saved columns: {list(df.columns)}")
    print(f"DEBUG - Lives saved sample: {df.head(2).to_dict('records')}")
    
    # Flexible column name detection for different data sources
    lives_saved_col = None
    for col in ['lives_saved', 'Lives_Saved', 'deaths_prevented', 'deaths_averted']:
        if col in df.columns:
            lives_saved_col = col
            break
    
    year_col = None
    for col in ['year', 'Year', 'calendar_year']:
        if col in df.columns:
            year_col = col
            break
    
    # Validate required columns
    if lives_saved_col is None or year_col is None:
        print(f"WARNING: Required columns missing. Found: {list(df.columns)}")
        print(f"Looking for lives_saved column: {lives_saved_col}")
        print(f"Looking for year column: {year_col}")
        return create_base_figure("Lives Saved", "Required data columns missing")

    fig = create_base_figure(
        "Estimating: How Many Lives Do Measles Vaccines Save Each Year?",
        "United States • Simulated estimates using WHO EPI50 mathematical models (not real observed data)"
    )

    # Add subtle background gradient for visual depth
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        fillcolor=f"rgba(140, 29, 64, 0.05)",
        layer="below", line_width=0,
    )

    # ASU-branded color scale for data intensity visualization
    asu_colorscale = [
        [0, ASUConfig.PRIMARY_MAROON],    # Lower values
        [0.5, "#B8336A"],                # Intermediate values
        [1, ASUConfig.PRIMARY_GOLD]      # Higher values
    ]

    # Main visualization: Gradient-colored bars showing lives saved estimates
    fig.add_trace(go.Bar(
        x=df[year_col],
        y=df[lives_saved_col],
        name='Lives Saved Annually',
        marker=dict(
            color=df[lives_saved_col],        # Color intensity based on value
            colorscale=asu_colorscale,
            colorbar=dict(
                title=dict(
                    text="<b>Lives Saved</b>",
                    side="right",
                    font=dict(
                        size=ASUConfig.AXIS_TITLE_SIZE,
                        color=ASUConfig.DARK_GRAY,
                        family=ASUConfig.FONT_FAMILY
                    )
                ),
                tickformat=',.0f',            # Format as integers with commas
                tickfont=dict(
                    size=ASUConfig.AXIS_TICK_SIZE,
                    color=ASUConfig.DARK_GRAY
                ),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=ASUConfig.LIGHT_GRAY,
                borderwidth=1
            ),
            line=dict(color=ASUConfig.PRIMARY_MAROON, width=1)
        ),
        hovertemplate='<b>Year: %{x}</b><br><b>Lives Saved: %{y:,.0f}</b><extra></extra>'
    ))

    # Apply standardized axis styling
    fig = style_axes(fig, "Year", "Number of Lives Saved")
    fig.update_yaxes(tickformat=',.0f')      # Format Y-axis with comma separators
    fig.update_layout(showlegend=False)      # Hide redundant legend

    # Contextual explanation for epidemiologists
    context_explanation = (
        "Computer-simulated estimates showing theoretical deaths prevented each year "
        "if measles vaccines had never been introduced in the United States since 1974."
    )

    # Comprehensive source attribution with methodological reference
    sources_text = (
        "WHO EPI50 Mathematical Models (2024) • "
        '<a href="https://github.com/WorldHealthOrganization/epi50-vaccine-impact" target="_blank">Full Data</a> • '
        "Published in <i>The Lancet</i>"
    )

    # Important disclaimer about data nature
    disclaimer_text = "<i>Note: These are theoretical projections, not observed deaths</i>"

    fig = add_comprehensive_data_source(fig, {"Source": sources_text}, f"{context_explanation}<br><br>{disclaimer_text}")

    return fig

def create_index_page(output_dir="docs"):
    """
    Generate a comprehensive index page listing all available visualizations.
    
    This function creates an HTML landing page that serves as a navigation
    hub for all generated measles visualizations, with timestamps and
    professional styling consistent with the visualization framework.
    
    Parameters:
    -----------
    output_dir : str
        Directory where HTML files are stored
    """
    
    # Get current data refresh timestamp for transparency
    refresh_time = get_data_refresh_time()
    
    # Professional HTML template with ASU branding elements
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Measles Data Visualizations</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f8f8; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
        h1 {{ color: #8C1D40; text-align: center; }}
        .viz-list {{ list-style: none; padding: 0; }}
        .viz-list li {{ margin: 10px 0; }}
        .viz-list a {{ 
            display: block; padding: 15px; background: #f0f0f0; 
            text-decoration: none; color: #8C1D40; border-radius: 5px;
            border-left: 4px solid #8C1D40;
        }}
        .viz-list a:hover {{ background: #e0e0e0; }}
        .last-updated {{ text-align: center; color: #666; font-size: 0.9em; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Measles Data Visualizations</h1>
        <p>Click any link below to view the standalone visualization:</p>
        
        <ul class="viz-list">
            <li><a href="measles_timeline.html">Historical Timeline</a></li>
            <li><a href="recent_trends.html">Recent Trends (2015-2025)</a></li>
            <li><a href="disease_contagiousness.html">Disease Contagiousness Comparison</a></li>
            <li><a href="us_measles_map.html">US State Map</a></li>
            <li><a href="lives_saved.html">Lives Saved by Vaccination</a></li>
        </ul>
        
        <div class="last-updated">
            Data last refreshed: {refresh_time}
        </div>
    </div>
</body>
</html>"""
    
    # Save index page with error handling
    index_path = Path(output_dir) / "index.html"
    try:
        with open(index_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ Created {index_path}")
    except Exception as e:
        print(f"ERROR creating index.html: {e}")

# =============================================================================
# MAIN EXECUTION AND ORCHESTRATION
# =============================================================================

def main():
    """
    Main execution function that orchestrates the complete visualization pipeline.
    
    This function coordinates data loading, visualization generation, and output
    file creation with comprehensive error handling and progress reporting.
    Epidemiologists can modify this function to customize the generation workflow
    or add additional visualization types.
    
    Returns:
    --------
    bool
        True if at least one visualization was successfully generated
    """

    print(f"Starting visualization generation at {datetime.now()}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Validate Python environment and required dependencies
    try:
        import plotly
        print(f"✓ Plotly version: {plotly.__version__}")
    except ImportError:
        print("ERROR: Plotly not installed. Run: pip install plotly")
        return False
    
    try:
        import requests
        print(f"✓ Requests available")
    except ImportError:
        print("ERROR: Requests not installed. Run: pip install requests")
        return False

    # Display current working environment for troubleshooting
    print(f"Contents of current directory: {os.listdir('.')}")
    
    # Verify and create data directory structure
    if os.path.exists('data'):
        print(f"Contents of data directory: {os.listdir('data')}")
        if os.path.exists('data/backups'):
            print(f"Contents of backup directory: {os.listdir('data/backups')}")
    else:
        print("WARNING: No data directory found! Creating...")
        os.makedirs('data', exist_ok=True)

    # Load all required datasets
    print("Loading data...")
    data = load_data()
    if data is None:
        print("Failed to load data. Please check file paths and connections.")
        return False

    print("Creating visualizations...")
    
    # Display data loading summary for verification
    for key, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"  {key}: {len(df)} rows, {len(df.columns)} columns")
            if len(df) > 0:
                print(f"    Sample columns: {list(df.columns)[:5]}")
            else:
                print(f"    DataFrame is empty!")
        else:
            print(f"  {key}: {type(df)}")

    # Define visualization pipeline with error handling
    visualizations = [
        ("Measles Timeline", "measles_timeline", lambda: create_measles_timeline(data['timeline'])),
        ("Recent Trends", "recent_trends", lambda: create_recent_trends(data['usmeasles'], data['mmr'])),
        ("Disease Contagiousness", "disease_contagiousness", lambda: create_rnaught_comparison()),
        ("US Map", "us_measles_map", lambda: create_us_map(data['usmap'])),
        ("Lives Saved", "lives_saved", lambda: create_lives_saved_chart(data['vaccine_impact']))
    ]
    
    successful_saves = 0
    
    # Generate each visualization with individual error handling
    for viz_name, filename, create_func in visualizations:
        try:
            print(f"- Creating {viz_name}")
            fig = create_func()
            save_figure(fig, filename)
            successful_saves += 1
        except Exception as e:
            print(f"ERROR creating {viz_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback error visualization to maintain pipeline integrity
            try:
                error_fig = create_base_figure(viz_name, f"Error: {str(e)}")
                save_figure(error_fig, filename)
            except:
                print(f"Could not create error fallback for {viz_name}")

    # Generate navigation index page
    try:
        create_index_page()
        print("✓ Created index page")
    except Exception as e:
        print(f"ERROR creating index page: {e}")

    # Final execution summary and file verification
    print(f"\nVisualization generation completed at {datetime.now()}")
    print(f"Successfully created {successful_saves} out of {len(visualizations)} visualizations")
    
    # Verify output directory and file sizes
    docs_path = Path('docs')
    if docs_path.exists():
        files = list(docs_path.glob('*.html'))
        print(f"Files in docs directory: {[f.name for f in files]}")
        
        # Display file sizes for monitoring and troubleshooting
        for file in files:
            size = file.stat().st_size
            print(f"  {file.name}: {size:,} bytes")
    else:
        print("ERROR: docs directory was not created!")
        return False
        
    return successful_saves > 0

# Entry point for direct script execution
if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)  # Exit with error code if generation failed
