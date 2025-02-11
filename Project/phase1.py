# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable

df = pd.read_csv('./assets/application_record.csv')

font_title = {'family': 'serif', 'color': 'blue', 'size': 20}
font_title_big = {'family': 'serif', 'color': 'blue', 'size': 30}
font_label = {'family': 'serif', 'color': 'darkred', 'size': 15}

df_cleaned = df.copy()

x = PrettyTable()
x.field_names = df_cleaned.columns.tolist()
for index, row in df_cleaned.head(100).iterrows():
    x.add_row(row.tolist())

print(x.get_string(title='Credit card approval record Data'))

# %%
# ==============================================================================
# Data Cleaning
# ==============================================================================

# 1. missing value fillna
print("The number of missing values in OCCUPATION_TYPE:", df_cleaned['OCCUPATION_TYPE'].isnull().sum())
df_cleaned['OCCUPATION_TYPE'] = df_cleaned['OCCUPATION_TYPE'].fillna('Unknown')
print("The number of missing values in OCCUPATION_TYPE after fillna:", df_cleaned['OCCUPATION_TYPE'].isnull().sum())

# 2. create new columns from day_birth
df_cleaned['AGE'] = (df_cleaned['DAYS_BIRTH'] / -365).round()
df_cleaned['AGE_GROUP'] = pd.cut(df_cleaned['AGE'],
                                 bins=[0, 29, 39, 49, 59, 69],
                                 labels=["20s", "30s", "40s", "50s", "60s"])

# 3. IQR
q1 = df_cleaned['CNT_CHILDREN'].quantile(0.25)
q3 = df_cleaned['CNT_CHILDREN'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df_cleaned = df_cleaned[
    (df_cleaned['CNT_CHILDREN'] >= lower_bound) & (df_cleaned['CNT_CHILDREN'] <= upper_bound)]

# 4. 1% upper data remove
amt_income_upper_limit = df_cleaned['AMT_INCOME_TOTAL'].quantile(0.99)
df_cleaned = df_cleaned[df_cleaned['AMT_INCOME_TOTAL'] < amt_income_upper_limit]

# 5. if datas_employed
df_cleaned['DAYS_EMPLOYED'] = df_cleaned['DAYS_EMPLOYED'].apply(lambda x: None if x > 0 else x)

# %%
print(df_cleaned.info())
pd.set_option('display.max_columns', None)
print(df_cleaned.describe(include='all').round(2))

# %%
# ==============================================================================
# PCA
# ==============================================================================
numeric_df = df_cleaned[['CNT_CHILDREN', 'AGE', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS']]

# fillna with mean
numeric_df.loc[:, :] = numeric_df.fillna(numeric_df.mean())

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_df)

pca = PCA()
pca_result = pca.fit(numeric_scaled)

singular_values = pca.singular_values_
print("Singular Values:", np.round(singular_values, 2))

condition_number = max(singular_values) / min(singular_values)

print("Condition Number:", round(condition_number, 2))

explained_variance = np.round(pca.explained_variance_ratio_, 2)
print("Explained Variance Ratio of Each Principal Component", explained_variance)

cumulative_variance = np.round(np.cumsum(explained_variance), 2)
print("Cumulative Explained Variance Ratio:", cumulative_variance)

n_components = np.argmax(cumulative_variance >= 0.80) + 1
print(f"Number of Principal Components to Achieve 80% Explained Variance: {n_components}")

pca_optimal = PCA(n_components=n_components)
pca_result = pca_optimal.fit_transform(numeric_scaled)

pca_df = pd.DataFrame(data=pca_result, columns=[f"PC{i + 1}" for i in range(n_components)])

explained_variance_optimal = pca_optimal.explained_variance_ratio_
print("Explained Variance Ratio of Selected Principal Components:", np.round(explained_variance_optimal, 2))

components = pd.DataFrame(pca_optimal.components_, columns=numeric_df.columns,
                          index=[f"PC{i + 1}" for i in range(n_components)])

print("Components of Selected Principal Components:\n", components)

# %%
# Normality Test ( Shapiro-Wilk Test )
from scipy.stats import shapiro

df_sampled = df_cleaned.sample(n=500, random_state=42)
# Perform the Shapiro-Wilk test
stat, p = shapiro(df_sampled['AMT_INCOME_TOTAL'])

print(f"Test Statistic: {stat:.2f}")
print(f"P-Value: {p:.2f}")
# Observations
if p > 0.05:
    print("Gaussian distribution")
else:
    print("Noe Gaussian distribution")

stats.probplot(df_sampled['AMT_INCOME_TOTAL'], dist="norm", plot=plt)
plt.title("Q-Q Plot of Income Data")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Values")
plt.show()

# %%
# Data transformation

df_cleaned['AMT_INCOME_TOTAL_log'] = np.log(df_cleaned['AMT_INCOME_TOTAL'])

# Plot the distributions
fig, axs = plt.subplots(1, 2, figsize=(15, 8))

stats.probplot(df_cleaned['AMT_INCOME_TOTAL'], dist="norm", plot=axs[0])
axs[0].set_title('Original')

stats.probplot(df_cleaned['AMT_INCOME_TOTAL_log'], dist="norm", plot=axs[1])
axs[1].set_title('Log Transformation')
plt.tight_layout()
plt.show()

# %%
# Heatmap & Pearson correlation coefficient matrix
selected_columns = ['AGE', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED']
correlation_matrix = df_cleaned[selected_columns].corr()

print(correlation_matrix.round(2))

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGn", fmt=".2f")
plt.title("Correlation Heatmap of Selected Features", fontdict=font_title)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=45)
plt.xlabel("Features", fontdict=font_label)
plt.ylabel("Features", fontdict=font_label)
plt.tight_layout()
plt.show()

# Scatter plot matrix
sns.pairplot(df_cleaned[selected_columns], kind="scatter", diag_kind='kde')
plt.suptitle("Scatter Plot Matrix of Selected Features", fontdict=font_title)
plt.subplots_adjust(top=0.95)
plt.show()

# %%
# Multivariate Kernel Density Estimate
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df_cleaned[["AGE", "AMT_INCOME_TOTAL"]],
    x="AGE",
    y="AMT_INCOME_TOTAL",
)
plt.title("Multivariate Kernel Density Estimate of Age and Income", fontdict=font_title)
plt.xlabel("Age", fontdict=font_label)
plt.ylabel("Income", fontdict=font_label)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
columns_of_interest = ["AGE", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "DAYS_EMPLOYED"]
statistics = df_cleaned[columns_of_interest].describe()

print(statistics.round(2))

# ==============================================================================
# Graphs
# ==============================================================================
# %%
# Line-plot
line_plot = df_cleaned.groupby('AGE', observed=False)['AMT_INCOME_TOTAL'].mean()

plt.figure(figsize=(10, 6))
plt.plot(line_plot.index, line_plot.values, marker='o', color='purple', linewidth=4)
plt.title("Average Income by AGE", fontdict=font_title)
plt.xlabel("AGE", fontdict=font_label)
plt.ylabel("Average Income", fontdict=font_label)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Bar plot: stack, group
income_gender = df_cleaned.groupby(['NAME_INCOME_TYPE', 'CODE_GENDER'], observed=False).size().unstack()
income_gender.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Income Type by Gender", fontdict=font_title)
plt.xlabel("Income Type", fontdict=font_label)
plt.ylabel("Count", fontdict=font_label)
plt.legend(title="Gender")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='NAME_EDUCATION_TYPE', order=df_cleaned['NAME_EDUCATION_TYPE'].value_counts().index)
plt.title("Count of EDUCATION_TYPE", fontdict=font_title)
plt.xlabel("EDUCATION_TYPE", fontdict=font_label)
plt.ylabel("Count", fontdict=font_label)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Pie chart
gender_counts = df_cleaned['CODE_GENDER'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=['Female', 'Male'], autopct='%1.1f%%', textprops={'fontsize': 15},
        colors=['salmon', 'skyblue'], startangle=90, explode=(0, 0.1))
plt.title("Gender Distribution", fontdict=font_title)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Dist plot
plt.figure(figsize=(10, 6))
sns.distplot(df_cleaned['AMT_INCOME_TOTAL'], kde=True, bins=20, color='purple')
plt.title("Income Distribution", fontdict=font_title)
plt.xlabel("Income", fontdict=font_label)
plt.ylabel("Count", fontdict=font_label)
plt.grid()
plt.tight_layout()
plt.show()
# %%
# Pair plot

df_pairplot = df_cleaned[['AGE', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', "AGE_GROUP"]]
sns.pairplot(data=df_pairplot, palette="coolwarm", hue="AGE_GROUP")
plt.suptitle("Pair Plot", fontsize=20, color='blue')  # suptitle의 y 위치 조정
plt.subplots_adjust(top=0.91)  # 상단 여백을 조정하여 suptitle과 범례가 겹치지 않게 함
plt.show()

# %%
# Heatmap with cbar
selected_columns = ['AGE', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED']
plt.figure(figsize=(10, 10))
correlation_matrix_selected = df_cleaned[selected_columns].corr()
sns.heatmap(correlation_matrix_selected, annot=True, cbar=True, cmap="coolwarm", fmt=".2f")
plt.xticks(rotation=45)
plt.title("Correlation Heatmap of Selected Features", fontdict=font_title)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.xlabel("Features", fontdict=font_label)
plt.ylabel("Features", fontdict=font_label)
plt.tight_layout()
plt.show()

# %%
# Histogram plot with KDE
plt.figure(figsize=(8, 6))
sns.histplot(df_cleaned['AMT_INCOME_TOTAL'], kde=True, bins=30, color='blue')
plt.title("Histogram with KDE of Total Income", fontdict=font_title)
plt.xlabel("AMT_INCOME_TOTAL", fontdict=font_label)
plt.ylabel("Frequency", fontdict=font_label)
plt.tight_layout()
plt.show()

# %%
# QQ-plot
plt.figure(figsize=(8, 6))
stats.probplot(df_cleaned['AMT_INCOME_TOTAL'], dist="norm", plot=plt)
plt.title("QQ-plot of Income Data", fontdict=font_title)
plt.xlabel("Theoretical Quantiles", fontdict=font_label)
plt.ylabel("Ordered Values", fontdict=font_label)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# KDE plot will fill, alpha = 0.6, chose a palette, chose a linewidth
sns.set_palette("dark")
plt.figure(figsize=(8, 6))
sns.kdeplot(df_cleaned['AMT_INCOME_TOTAL'], fill=True, alpha=0.6, linewidth=2)
plt.title("KDE Plot of Income Data", fontdict=font_title)
plt.xlabel("Income", fontdict=font_label)
plt.ylabel("Density", fontdict=font_label)
plt.grid(True)
plt.show()

# %%
# Im or reg plot with scatter representation and regression line
plt.figure(figsize=(8, 6))
sns.regplot(data=df_cleaned, x="AGE", y="AMT_INCOME_TOTAL", scatter_kws={"color": "purple"},
            line_kws={"color": "red"})
plt.title("Scatter Plot with Regression Line using regplot", fontdict=font_title)
plt.xlabel("AGE", fontdict=font_label)
plt.ylabel("Income", fontdict=font_label)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Multivariate Box or Boxen plot
plt.figure(figsize=(12, 6))
sns.boxenplot(data=df_cleaned, x="CNT_CHILDREN", y="AMT_INCOME_TOTAL", hue="AGE_GROUP")
plt.title("Boxen Plot of Income by Number of Children and Age Group", fontdict=font_title)
plt.xlabel("Number of Children", fontdict=font_label)
plt.ylabel("Income", fontdict=font_label)
plt.legend(title="Age Group", loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Area plot
area_data = df_cleaned.groupby('AGE_GROUP', observed=False)['AMT_INCOME_TOTAL'].sum()
plt.figure(figsize=(10, 6))
area_data.plot(kind='area', color='skyblue', alpha=0.6)
plt.title("Total Income by Age Group", fontdict=font_title)
plt.xlabel("Age Group", fontdict=font_label)
plt.ylabel("Total Income", fontdict=font_label)
plt.grid(True)
plt.show()

# %%
# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_cleaned, x="AGE_GROUP", y="AMT_INCOME_TOTAL", palette="muted", hue="AGE_GROUP")
plt.title("Income Distribution by Age Group", fontdict=font_title)
plt.xlabel("Age Group", fontdict=font_label)
plt.ylabel("Income", fontdict=font_label)
plt.grid(True)
plt.show()

# %%
# Joint plot with KDE and scatter representation

sns.jointplot(data=df_cleaned, x="AGE", y="AMT_INCOME_TOTAL", hue="CODE_GENDER", kind="scatter").fig.suptitle(
    "Joint Plot of Age vs Income by Gender", fontdict=font_title)
plt.xlabel("AGE", fontdict=font_label)
plt.ylabel("Income", fontdict=font_label)
plt.legend(title="Gender", loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# %%
# Rug plot
plt.figure(figsize=(8, 4))
sns.kdeplot(data=df_cleaned, x="AMT_INCOME_TOTAL", fill=True, color="red", alpha=0.5)
sns.rugplot(data=df_cleaned, x="AMT_INCOME_TOTAL", height=0.1, color="blue")
plt.title("KDE and Rug Plot of Income", fontdict=font_title)
plt.xlabel("Income", fontdict=font_label)
plt.ylabel("Density", fontdict=font_label)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# 3D plot and contour plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
x = df_cleaned['AGE']
y = df_cleaned['DAYS_EMPLOYED']
z = df_cleaned['AMT_INCOME_TOTAL']
ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', alpha=0.6)
ax.set_xlabel('Age', fontdict=font_label)
ax.set_ylabel('Days Employed', fontdict=font_label)
ax.set_zlabel('Income', fontdict=font_label)
plt.title('3D Scatter Plot', fontdict=font_title)
plt.tight_layout()
plt.show()

# %%
# Cluster map
df_cluster = df_cleaned[['AGE', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL']]
df_cluster = df_cluster.dropna()
clustermap = sns.clustermap(df_cluster.corr(), cmap="coolwarm", annot=True, linewidths=.5)
plt.subplots_adjust(top=0.9)
clustermap.fig.suptitle("Cluster Map of Selected Features", fontsize=20, color='blue')
plt.tight_layout()
plt.show()

# %%
# Hexbin
plt.figure(figsize=(10, 6))
plt.hexbin(df_cleaned['AGE'], df_cleaned['AMT_INCOME_TOTAL'], gridsize=30, cmap='viridis')
plt.colorbar(label='Count in Bin')
plt.xlabel("Age", fontdict=font_label)
plt.ylabel("Income", fontdict=font_label)
plt.title("Hexbin Plot of Age vs Income", fontdict=font_title)
plt.grid(True)
plt.show()

# %%
# Strip plot
plt.figure(figsize=(10, 6))
sns.stripplot(data=df_cleaned, x="AGE_GROUP", y="AMT_INCOME_TOTAL", hue="AGE_GROUP", palette="muted", dodge=True,
              jitter=True)
plt.title("Income Distribution by Age Group (Strip Plot)", fontdict=font_title)
plt.xlabel("AGE_GROUP", fontdict=font_label)
plt.ylabel("Income ", fontdict=font_label)
plt.grid(True)
plt.show()

# %%
# Swarm plot
df_sampled = df_cleaned.sample(50000)
plt.figure(figsize=(10, 6))
sns.swarmplot(data=df_sampled, x="AGE_GROUP", y="AMT_INCOME_TOTAL", hue="AGE_GROUP", palette="muted", dodge=False,
              legend=False, size=0.3)
plt.title("Income Distribution by AGE_GROUP (Swarm Plot)", fontdict=font_title)
plt.xlabel("AGE_GROUP", fontdict=font_label)
plt.ylabel("Income", fontdict=font_label)
plt.grid(True)
plt.show()

# %%
# subplots  1
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# 1. AGE_GROUP vs AMT_INCOME_TOTAL(Bar Chart)
sns.barplot(
    data=df_cleaned,
    x="AGE_GROUP",
    y="AMT_INCOME_TOTAL",
    ax=axs[0, 0],
    hue="CODE_GENDER",
    palette="muted"
)
axs[0, 0].set_title("Average Income by Age Group and Gender", fontdict=font_title)
axs[0, 0].set_xlabel("Age Group", fontdict=font_label)
axs[0, 0].set_ylabel("Income", fontdict=font_label)

# 2. FLAG_OWN_CAR vs AMT_INCOME_TOTAL (Pie Chart)
car_ownership = df_cleaned.groupby('FLAG_OWN_CAR')['AMT_INCOME_TOTAL'].mean()
axs[0, 1].pie(
    car_ownership,
    labels=car_ownership.index,
    autopct='%1.1f%%',
    colors=sns.color_palette("colorblind"),
    startangle=140,
)
axs[0, 1].legend(
    loc="upper left",
    title="Car Ownership",
    bbox_to_anchor=(1, 0.9),
    fontsize=10
)
axs[0, 1].set_title("Car Ownership Distribution by Income", fontdict=font_title)

# 3. Income Type Distribution (Bar Chart)
income_type_counts = df_cleaned['NAME_INCOME_TYPE'].value_counts()
axs[1, 0].bar(
    income_type_counts.index,
    income_type_counts.values,
    color=sns.color_palette("colorblind"),
)
axs[1, 0].set_title("Income Type Distribution", fontdict=font_title)
axs[1, 0].set_xlabel("Income Type", fontdict=font_label)
axs[1, 0].set_ylabel("Count", fontdict=font_label)
axs[1, 0].tick_params(axis='x', rotation=30)

# 4. Realty Ownership vs Income (Box Plot)
sns.boxplot(
    data=df,
    x="FLAG_OWN_REALTY",
    y="AMT_INCOME_TOTAL",
    ax=axs[1, 1],
    palette="pastel",
    hue="FLAG_OWN_CAR"
)
axs[1, 1].set_title("Income Distribution by Realty Ownership", fontdict=font_title)
axs[1, 1].set_xlabel("Owns Realty", fontdict=font_label)
axs[1, 1].set_ylabel("Income", fontdict=font_label)
plt.tight_layout()
plt.show()

# %%
# subplots  2

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

sns.barplot(
    data=df_cleaned,
    x="CNT_CHILDREN",
    y="AMT_INCOME_TOTAL",
    ax=axs[0, 0],
    palette="muted",
    hue="CNT_CHILDREN"
)
axs[0, 0].set_title("Average Income by Number of Children", fontdict=font_title)
axs[0, 0].set_xlabel("Number of Children", fontdict=font_label)
axs[0, 0].set_ylabel("Average Income", fontdict=font_label)
axs[0, 0].legend().set_visible(False)

children_income = df_cleaned.groupby('CNT_CHILDREN')['AMT_INCOME_TOTAL'].mean()
axs[0, 1].pie(
    children_income,
    labels=children_income.index,
    autopct='%1.1f%%',
    colors=sns.color_palette("deep"),
)
axs[0, 1].legend(
    loc="upper left",
    fontsize=10
)
axs[0, 1].set_title("Number of Children Distribution by Income", fontdict=font_title)

sns.barplot(
    data=df_cleaned,
    x="FLAG_OWN_REALTY",
    y="AMT_INCOME_TOTAL",
    ax=axs[1, 0],
    hue="FLAG_OWN_REALTY",
    palette="husl"
)
axs[1, 0].set_title("Income Distribution by Realty Ownership", fontdict=font_title)
axs[1, 0].set_xlabel("Realty Ownership", fontdict=font_label)
axs[1, 0].set_ylabel("Income", fontdict=font_label)

own_realty = df_cleaned.groupby('FLAG_OWN_REALTY')['AMT_INCOME_TOTAL'].mean()
axs[1, 1].pie(
    own_realty,
    labels=own_realty.index,
    autopct='%1.1f%%',
    colors=sns.color_palette("colorblind6"),
)
axs[1, 1].legend(
    loc="upper left",
    fontsize=10
)
axs[1, 1].set_title("Realty Ownership Distribution by Income", fontdict=font_title)

plt.tight_layout()
plt.show()

# %% subplots  3
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

sns.barplot(
    data=df_cleaned,
    x="FLAG_OWN_CAR",
    y="AMT_INCOME_TOTAL",
    ax=axs[0, 0],
    hue="FLAG_OWN_CAR",
    palette="dark"
)
axs[0, 0].set_title("Average Income by Car Ownership", fontdict=font_title)
axs[0, 0].set_xlabel("Car Ownership", fontdict=font_label)
axs[0, 0].set_ylabel("Average Income", fontdict=font_label)

own_car = df_cleaned.groupby('FLAG_OWN_CAR')['AMT_INCOME_TOTAL'].mean()
axs[0, 1].pie(
    own_car,
    labels=own_car.index,
    autopct='%1.1f%%',
    colors=sns.color_palette("deep"),
)
axs[0, 1].legend(
    loc="upper left",
    fontsize=10
)
axs[0, 1].set_title("Car Ownership Distribution by Income", fontdict=font_title)

sns.barplot(
    data=df_cleaned,
    x="NAME_FAMILY_STATUS",
    y="AMT_INCOME_TOTAL",
    ax=axs[1, 0],
    hue="NAME_FAMILY_STATUS",
    palette="colorblind"
)
axs[1, 0].set_title("Average Income by Family Status", fontdict=font_title)
axs[1, 0].set_xlabel("Family Status", fontdict=font_label)
axs[1, 0].set_ylabel("Average Income", fontdict=font_label)

family_status = df_cleaned.groupby('NAME_FAMILY_STATUS')['AMT_INCOME_TOTAL'].mean()
axs[1, 1].pie(
    family_status,
    labels=family_status.index,
    autopct='%1.1f%%',
    colors=sns.color_palette("colorblind6"),
)
axs[1, 1].legend(
    loc="upper left",
    fontsize=10,
    bbox_to_anchor=(1, 1)
)
axs[1, 1].set_title("Family Status Distribution by Income", fontdict=font_title)

plt.tight_layout()
plt.show()

# %% subplots  4

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

sns.barplot(
    data=df_cleaned,
    x="NAME_HOUSING_TYPE",
    y="AMT_INCOME_TOTAL",
    ax=axs[0, 0],
    hue="NAME_HOUSING_TYPE",
    palette="dark"
)
axs[0, 0].set_title("Average Income by Housing Type", fontdict=font_title)
axs[0, 0].set_xticks(range(len(axs[0, 0].get_xticklabels())))
axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45)
axs[0, 0].set_xlabel("Housing Type", fontdict=font_label)
axs[0, 0].set_ylabel("Average Income", fontdict=font_label)

housing_type = df_cleaned.groupby('NAME_HOUSING_TYPE')['AMT_INCOME_TOTAL'].mean()
axs[0, 1].pie(
    housing_type,
    labels=housing_type.index,
    autopct='%1.1f%%',
    colors=sns.color_palette("deep"),
)
axs[0, 1].legend(
    loc="upper left",
    fontsize=10,
    bbox_to_anchor=(1, 1)
)
axs[0, 1].set_title("Housing Type Distribution by Income", fontdict=font_title)

sns.barplot(
    data=df_cleaned,
    x="AGE",
    y="NAME_FAMILY_STATUS",
    ax=axs[1, 0],
    hue="NAME_FAMILY_STATUS",
    palette="colorblind"
)
axs[1, 0].set_title("Family Status by Age", fontdict=font_title)
axs[1, 0].set_xlabel("Age", fontdict=font_label)
axs[1, 0].set_ylabel("Family Status", fontdict=font_label)

ageGroup = df_cleaned['AGE_GROUP'].value_counts()
axs[1, 1].pie(
    ageGroup,
    labels=ageGroup.index,
    autopct='%1.1f%%',
    colors=sns.color_palette("colorblind6"),
)
axs[1, 1].legend(
    loc="upper left",
    fontsize=10,
    bbox_to_anchor=(1, 1)
)
axs[1, 1].set_title("Age Group Distribution", fontdict=font_title)

plt.tight_layout()
plt.show()
