import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 0)

res_ungrouped_df = pd.read_csv('data/resources.csv')
# res_ungrouped_df = res_ungrouped_df[:1000]


def group_by_res_id(df):
    desc_most_exp_item = df.loc[df.price == max(df.price)].iloc[0]['description']
    return pd.Series(dict(id=df['id'].iloc[0],
                          price='{:.2f}'.format(df['price'].sum()),
                          quantity=df['quantity'].sum(),
                          description=desc_most_exp_item))


res_df = res_ungrouped_df.groupby('id').apply(group_by_res_id).reset_index(drop=True)
res_df = res_df[['id', 'quantity', 'price', 'description']]

print(res_df.head())

res_df.to_csv('data/resources_grouped.csv', index=False)