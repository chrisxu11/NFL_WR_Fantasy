import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


seasons = list(range(2024, 2025))
seasonal_data = nfl.import_seasonal_data(seasons, s_type='REG')

# Get roster data to filter for WRs
rosters = nfl.import_seasonal_rosters(seasons)

# Merge with roster data to get positions
wr_data = seasonal_data.merge(
    rosters[['player_id', 'position', 'player_name']].drop_duplicates('player_id'),
    on='player_id',
    how='left'
)
wr_data['player_name'] = wr_data['player_name'].combine_first(wr_data['player_name'])
wr_data = wr_data[wr_data['position'] == 'WR'].copy()


# Calculate PPR fantasy points
wr_data['ppr_points'] = (
    wr_data['receiving_yards'] * 0.1 +
    wr_data['receiving_tds'] * 6 +
    wr_data['receptions'] * 1
)
    
# Filter for recent season and minimum targets
wr_recent = wr_data[(wr_data['season'] == 2024) & (wr_data['targets'] >= 30)].copy()
    
# Calculate additional metrics
wr_recent['points_per_game'] = wr_recent['ppr_points'] / wr_recent['games']
    
# Create rankings
metrics = ['ppr_points', 'target_share', 'points_per_game']
for metric in metrics:
    wr_recent[f'{metric}_rank'] = wr_recent[metric].rank(ascending=False)
    
# Create a composite score based off the ppr points (points per reception), target share, and points per game 
wr_recent['composite_score'] = (
    wr_recent['ppr_points_rank'] * 0.4 +
    wr_recent['target_share_rank'] * 0.3 +
    wr_recent['points_per_game_rank'] * 0.3
)
    
# Final ranking
wr_recent['final_rank'] = wr_recent['composite_score'].rank()
    

top_wrs = wr_recent.sort_values('final_rank').head(20)


top_50_wrs = wr_recent.sort_values('final_rank').head(50)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('WR Fantasy Football Analysis (2024 Season)', fontsize=14, fontweight='bold')

# 1. Top 10 WRs by PPR Points
top_10_ppr = top_wrs.nlargest(10, 'ppr_points')
bars1 = axes[0, 0].barh(top_10_ppr['player_name'], top_10_ppr['ppr_points'])
axes[0, 0].set_xlabel('PPR Points')
axes[0, 0].set_title('Top 10 WRs by Total PPR Points')
axes[0, 0].invert_yaxis()
# Add value labels on bars
for bar in bars1:
    width = bar.get_width()
    axes[0, 0].text(width + 5, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1f}', ha='left', va='center')

# 2. Top 10 WRs by Points Per Game
top_10_ppg = top_wrs.nlargest(10, 'points_per_game')
bars2 = axes[0, 1].barh(top_10_ppg['player_name'], top_10_ppg['points_per_game'])
axes[0, 1].set_xlabel('Points Per Game')
axes[0, 1].set_title('Top 10 WRs by Points Per Game')
axes[0, 1].invert_yaxis()
# Add value labels on bars
for bar in bars2:
    width = bar.get_width()
    axes[0, 1].text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1f}', ha='left', va='center')

# 3. Target Share vs PPR Points
scatter = axes[1, 0].scatter(wr_recent['target_share'], wr_recent['ppr_points'], 
                            alpha=0.6, s=80)
axes[1, 0].set_xlabel('Target Share')
axes[1, 0].set_ylabel('PPR Points')
axes[1, 0].set_title('Target Share vs PPR Points')

# Add trendline
z = np.polyfit(wr_recent['target_share'], wr_recent['ppr_points'], 1)
p = np.poly1d(z)
axes[1, 0].plot(wr_recent['target_share'], p(wr_recent['target_share']), "r--", alpha=0.8)

# Label top 5 players
for i, row in top_wrs.head(5).iterrows():
    axes[1, 0].annotate(row['player_name'], 
                       (row['target_share'], row['ppr_points']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)

# 4. Top 20 WRs by Composite Score (Horizontal Bar Chart in lower-right subplot)
top_20_composite = top_50_wrs.head(20).sort_values('composite_score', ascending=True)
ax = axes[1, 1]
bars = ax.barh(top_20_composite['player_name'], top_20_composite['composite_score'], color='#4C72B0')
ax.set_xlabel('Composite Score (Lower is Better)')
ax.set_title('Top 20 WRs by Composite Score')
ax.invert_yaxis()  # Best player at the top

# Add value labels on bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', ha='left', va='center', fontsize=8)

# Add some context to the composite score
fig.text(0.5, 0.02, 
         'Composite Score = (PPR Points Rank × 0.4) + (Target Share Rank × 0.3) + (Points Per Game Rank × 0.3)\nLower scores indicate better overall performance', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.1)
plt.savefig('combined_composite_score_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
# Create a separate table visualization
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('tight')
ax.axis('off')

# Prepare table data (ensure expected columns exist)
expected_cols = ['player_name', 'games', 'targets', 'target_share', 
                 'receptions', 'receiving_yards', 'receiving_tds', 
                 'ppr_points', 'points_per_game', 'final_rank']
for col in expected_cols:
    if col not in top_wrs.columns:
        top_wrs[col] = np.nan
table_data = top_wrs[expected_cols].copy()
table_data.columns = ['Player','Games', 'Targets', 'Target Share', 
                     'Receptions', 'Yards', 'TDs', 'PPR Points', 'PPG', 'Rank']
table_data = table_data.sort_values('Rank')

# Create table
table = ax.table(cellText=table_data.round(2).values,
                colLabels=table_data.columns,
                cellLoc='center',
                loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Highlight the top row (header)
for i in range(len(table_data.columns)):
    table[(0, i)].set_facecolor('#4C72B0')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors for readability
for i in range(1, len(table_data) + 1):
    if i % 2 == 0:
        for j in range(len(table_data.columns)):
            table[(i, j)].set_facecolor('#f0f0f0')

plt.title('Top 20 WRs for PPR Leagues (2024)', fontsize=14, fontweight='bold', pad=20)
plt.savefig('wr_analysis_table.png', dpi=300, bbox_inches='tight')
plt.show()

# Print correlation matrix
corr_matrix = wr_recent[['ppr_points', 'target_share', 'points_per_game', 
                        'receptions', 'receiving_yards', 'receiving_tds']].corr()

print("Correlation Matrix:")
print(corr_matrix.round(2))

top_50_ppr = wr_recent.nlargest(100, 'ppr_points')[['player_name', 'games', 'targets', 
                                                   'receptions', 'receiving_yards', 'receiving_tds', 
                                                   'ppr_points', 'points_per_game']].copy()

# Add ranking column
top_50_ppr['ppr_rank'] = range(1, len(top_50_ppr) + 1)

# Format the data for better display
top_50_ppr.columns = ['Player', 'Games', 'Targets', 'Receptions', 'Yards', 'TDs', 'PPR Points', 'PPG', 'Rank']

# Display the top 50 WRs
print("Top 50 WRs Based on PPR Points (2024 Season)")
print("=" * 90)
print(top_50_ppr.to_string(index=False))

# Create a visualization of the top 20 PPR scorers
plt.figure(figsize=(14, 10))
top_50_ppr = top_50_ppr.head(50).sort_values('PPR Points', ascending=True)
bars = plt.barh(top_50_ppr['Player'], top_50_ppr['PPR Points'])
plt.xlabel('PPR Points')
plt.title('Top 50 WRs by PPR Points (2024 Season)')


# Add value labels on bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 5, bar.get_y() + bar.get_height()/2, f'{width:.1f}', ha='left', va='center')

plt.tight_layout()
plt.savefig('top_50_ppr_wrs.png', dpi=300, bbox_inches='tight')

plt.show()
