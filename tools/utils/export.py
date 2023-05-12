import pandas as pd


def export_excel(info, output_path, order=None):
    df = pd.DataFrame(list(info))
    if order:
        df = df[order]
    wr = pd.ExcelWriter(output_path)
    df.to_excel(wr, encoding='utf-8', index=False)
    wr.save()
 