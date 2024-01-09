| ヘッダ名称   | 値例                                | データ型 | 説明                                    |
|--------------|-------------------------------------|---------|-----------------------------------------|
| created_at   | 2015-06-29                          | str     | データが作成された日付                   |
| tree_dbh     | 14                                  | int     | 木の胸高直径（DBH）                      |
| curb_loc     | OnCurb                              | str     | ツリーが路上にあるかどうか                |
| health       | 1                                   | int     | 木の健康状態（目的変数）                |
| steward      | 3or4                                | str     | 木の管理者やケアの数                     |
| guards       | Helpful                             | str     | 木を保護するためのガードの有無や種類    |
| sidewalk     | Damage                              | str     | 歩道の状態                              |
| user_type    | Volunteer                           | str     | データの記録者                          |
| problems     | StonesBranchLights                   | str     | 木に関連する問題                        |
| spc_common   | English oak                         | str     | 木の種類の一般的な名前                  |
| spc_latin    | Quercus robur                       | str     | 木のラテン名                            |
| nta          | QN45                                | str     | 地域（Neighborhood Tabulation Area）の略称 |
| nta_name     | Douglas Manor-Douglaston-Little Neck | str     | 地域（Neighborhood Tabulation Area）の正式な名称 |
| borocode     | 4                                   | int     | ニューヨーク市の行政区分（ボロー）のコード |
| boro_ct      | 4152901                             | int     | ニューヨーク市のセンサストラクトのコード  |
| boroname     | Queens                              | str     | ニューヨーク市の行政区分（ボロー）の名称  |
| zip_city     | Little Neck                         | str     | 郵便番号に関連する都市または地区        |
| cb_num       | 411                                 | int     | コミュニティボード番号                   |
| st_senate    | 11                                  | int     | 州上院地区番号                         |
| st_assem     | 25                                  | int     | 州議会地区番号                         |
| cncldist     | 23                                  | int     | 市議会の地区番号                       |


# `problems`カラムの処理について

`problems`カラムのユニーク
<details>
<summary>展開</summary>
'StonesBranchLights',
'Stones',
'BranchLights',
'StonesTrunkOther',
'StonesWiresRopeBranchLights',
'StonesWiresRope',
'WiresRope',
'MetalGrates',
'RootOtherTrunkOtherBranchOther',
'StonesTrunkOtherBranchOther',
'RootOther',
'TrunkOtherBranchLightsBranchOther',
'MetalGratesRootOtherBranchOther',
'StonesWiresRopeTrunkLightsBranchLights',
'MetalGratesRootOtherTrunkOther',
'StonesMetalGrates',
'StonesRootOtherBranchOther',
'StonesRootOther',
'BranchOther',
'TrunkOther',
'TrunkOtherBranchOther',
'StonesTrunkOtherBranchLightsBranchOther',
'StonesRootOtherWiresRopeTrunkOtherBranchLightsBranchOther',
'StonesBranchOther',
'WiresRopeBranchLights',
'RootOtherWiresRopeBranchOther',
'TrunkOtherBranchLights',
'RootOtherWiresRopeTrunkOtherBranchOther',
'StonesRootOtherTrunkOther',
'MetalGratesBranchOther',
'TrunkLightsBranchLights',
'RootOtherTrunkOther',
'StonesBranchLightsBranchOther',
'RootOtherBranchLights',
'StonesWiresRopeTrunkOther',
'WiresRopeBranchOther',
'MetalGratesTrunkOtherBranchOther',
'StonesWiresRopeTrunkOtherBranchLights',
'TrunkLights',
'RootOtherTrunkOtherBranchLightsBranchOther',
'StonesWiresRopeBranchLightsBranchOther',
'StonesMetalGratesWiresRopeTrunkLightsBranchLights',
RootOtherBranchOther',
'WiresRopeTrunkOtherBranchOther',
'RootOtherWiresRope',
'StonesTrunkOtherBranchLights',
'StonesTrunkLightsBranchLights',
'RootOtherWiresRopeBranchLightsBranchOther',
'StonesMetalGratesBranchLights',
'StonesRootOtherBranchLightsBranchOther',
'BranchLightsBranchOther',
'RootOtherWiresRopeTrunkLights',
'StonesWiresRopeBranchOther',
'TrunkLightsBranchLightsBranchOther',
'RootOtherWiresRopeTrunkOther',
'RootOtherWiresRopeTrunkOtherBranchLights',
'StonesRootOtherTrunkOtherBranchLights',
'WiresRopeTrunkOtherBranchLightsBranchOther',
'WiresRopeTrunkLightsBranchLights',
'RootOtherTrunkOtherBranchLights',
'StonesMetalGratesTrunkOther',
'RootOtherWiresRopeBranchLights',
'StonesRootOtherWiresRopeBranchLights',
'WiresRopeTrunkLights',
'StonesSneakers',
'MetalGratesWiresRope',
'StonesRootOtherWiresRopeBranchOther',
'MetalGratesTrunkOther',
'WiresRopeTrunkOther',
'RootOtherBranchLightsBranchOther',
'StonesRootOtherBranchLights',
'StonesRootOtherTrunkOtherBranchOther',
'RootOtherTrunkLightsBranchOther'

</details>

特徴量を生成する関数を作成した。

```
import pandas as pd

def transform_problem(raw_df:pd.DataFrame, problem_unique:set=None):
    """
    与えられたDataFrameに対して、problemsを分割して、各問題が含まれているかどうかを特徴量として追加する

    Parameters
    ----------
    raw_df : pd.DataFrame
        problemsを含むDataFrame
    problem_unique : set
        problemsに含まれる問題の集合(testデータを処理する場合には、trainデータの問題の集合を渡す)
    Returns
    -------
    pd.DataFrame
        problemsを分割して、各問題が含まれているかどうかを特徴量として追加したDataFrame
    problem_unique : set
        problemsに含まれる問題の集合(testデータを処理する場合にはNoneを返す)


    Examples
    --------
    >>> import pandas as pd
    >>> train = pd.read_csv("data/official/train.csv", index_col=0)
    >>> test = pd.read_csv("data/official/test.csv", index_col=0)
    >>> processed_train, problem_unique = transform_problem(train)
    >>> processed_test, _ = transform_problem(test, problem_unique)
    """
    def _split_string_by_uppercase(input_string):
        result = []
        current_word = ""
        if not type(input_string)==str:
            return result

        # Escape String "Other"
        for char in input_string:
            if char.isupper() and current_word:
                result.append(current_word)
                current_word = char
            else:
                current_word += char

        if current_word:
            result.append(current_word)

        return result

    df = raw_df.copy()
    df.loc[:,"problems_list"] = df.problems.apply(_split_string_by_uppercase)
    is_train = False
    if problem_unique is None:
        is_train = True
        problem_unique = set()
        for problems in df.loc[:,"problems"].unique():
            for problem in _split_string_by_uppercase(problems):
                problem_unique.add(problem)
    for unique_problem in problem_unique:
        df.loc[:,f"is_problem_{unique_problem}"] = df.loc[:,"problems_list"].apply(lambda x: unique_problem in x)
        df.loc[:,f"is_problem_{unique_problem}"] = df.loc[:,f"is_problem_{unique_problem}"].fillna(False)
    df.loc[:,"problems_count"] = df.loc[:,"problems_list"].apply(len)
    df = df.drop(columns=["problems_list"])
    df = df.drop(columns=["problems"])
    if is_train:
        return df, problem_unique
    else:
        return df, None
```

問題の数についてのヒストグラム
![image](https://github.com/rakawanegan/smbc_green_challenge/assets/79976928/7585a8f9-5661-4158-9a43-1d4e11d87d2c)

問題の数についての表
| problems | Count |
|----------|-------|
| 0        | 12243 |
| 2        | 2312  |
| 1        | 2219  |
| 3        | 1393  |
| 4        | 643   |
| 5        | 467   |
| 6        | 406   |
| 7        | 180   |
| 8        | 92    |
| 11       | 15    |
| 9        | 14    |

`problem`のユニークは以下の通り。

| 語句      | 説明                                               |
|----------|----------------------------------------------------|
| Branch   | 木の枝                           |
| Grates   | 金属製の格子状の物体、通気や遮蔽のために使用  |
| Lights   | ランプや照明器具               |
| Metal    | 金属の素材や物体                                   |
| Other    | ほかと組み合わせる               |
| Root     | 根っこ           |
| Rope     | 縄やロープ           |
| Sneakers | 運動靴、スポーツシューズ                             |
| Stones   | 小石や岩、地面に見られる堆積物                      |
| Trunk    | 木の主幹                                  |
| Wires    | 金属製の導線     |

木の部位に関係する名前はBranch、Root、Trunkの3つである。

木の部位ごとに問題があるかの特徴量を実装する
```
parts_list = [
    "Branch",
    "Trunk",
    "Root",
]
for parts in parts_list:
    train.loc[:,f"is_problem_{parts}"] = train.loc[:,"problems"].str.contains(parts)
```
訓練データにおける部位ごとの問題を表に示す。
| 問題          | Branch   | Trunk   | Root    |
|--------------|----------|---------|---------|
| is_problem   |          |         |         |
| False        | 3880     | 6255    | 6330    |
| True         | 3861     | 1486    | 1411    |


同様に部位以外の問題についても処理を行う。
```
cause_list = [
        'Grates',
        'Lights',
        'Metal',
        'Other',
        'Rope',
        'Sneakers',
        'Stones',
        'Wires'
        ]
for cause in cause_list:
    train.loc[:,f"is_cause_{cause}"] = train.loc[:,"problems"].str.contains(cause)
train.loc[:,[f"is_cause_{cause}" for cause in cause_list]] = train.loc[:,[f"is_cause_{cause}" for cause in cause_list]].fillna(False)
```
訓練データにおける問題内容を以下に示す。
|        | Grates | Lights | Metal | Other | Rope  | Sneakers | Stones | Wires |
|--------|--------|--------|-------|-------|-------|----------|--------|-------|
| False  | 19679  | 17268  | 19679 | 17026 | 19092 | 19974    | 15686  | 19092 |
| True   | 305    | 2716   | 305   | 2958  | 892   | 10       | 4298   | 892   |

察しの良い読者ならもう既に気づいていると思うが、この変換はOneHotエンコーディングと大文字を数えた特徴量の生成でしかない。「問題箇所」と「問題内容」を文字列を組み合わせた特徴量生成を考えている。


# LBのラベル比を算出する
```
precision = TP/(TP+FP)
recall = TP/(TP+FN)
```

データとして、以下を用いる
- 評価指標はマクロF1スコア
- すべて予測0でスコアは0.1003220
- すべて予測1でスコアは0.2940097
- すべて予測2でスコアは0.0218622

F1スコアはRecallとPrecisionの調和平均であり、ラベル比を表さない。
しかし、すべて同じ予測を行えば、
Recallはラベル比の逆数でありPrecisionは0となる。

つまりラベル比とF1スコアについて以下の式が成り立つ。
F1スコア＝3*調和平均（ラベル比，1）

これをラベル比について解くと、
ラベル比 ＝ 3*F1スコア/（2 ー 3*F1スコア）

これに各データを代入すれば、
ラベル0：0.177139〜18％
ラベル1：0.788955〜79％
ラベル2：0.033905〜3％

これは和がほぼ1であり解釈と一致する。

説明動画を見る限りLBとPBの比は公開されていなかった。
以後の計画として、提出するラベル比を調整して統計的にLBとPBの比を概算したい。


# データについて得られた知見（順次更新）
## カラム
| ヘッダ名称   | 値例                                | データ型 | 説明                                    |
|--------------|-------------------------------------|---------|-----------------------------------------|
| created_at   | 2015-06-29                          | str     | データが作成された日付                   |
| tree_dbh     | 14                                  | int     | 木の胸高直径（DBH）                      |
| curb_loc     | OnCurb                              | str     | ツリーが路上にあるかどうか                |
| health       | 1                                   | int     | 木の健康状態（目的変数）                |
| steward      | 3or4                                | str     | 木の管理者やケアの数                     |
| guards       | Helpful                             | str     | 木を保護するためのガードの有無や種類    |
| sidewalk     | Damage                              | str     | 歩道の状態                              |
| user_type    | Volunteer                           | str     | データの記録者                          |
| problems     | StonesBranchLights                   | str     | 木に関連する問題                        |
| spc_common   | English oak                         | str     | 木の種類の一般的な名前                  |
| spc_latin    | Quercus robur                       | str     | 木のラテン名                            |
| nta          | QN45                                | str     | 地域（Neighborhood Tabulation Area）の略称 |
| nta_name     | Douglas Manor-Douglaston-Little Neck | str     | 地域（Neighborhood Tabulation Area）の正式な名称 |
| borocode     | 4                                   | int     | ニューヨーク市の行政区分（ボロー）のコード |
| boro_ct      | 4152901                             | int     | ニューヨーク市のセンサストラクトのコード  |
| boroname     | Queens                              | str     | ニューヨーク市の行政区分（ボロー）の名称  |
| zip_city     | Little Neck                         | str     | 郵便番号に関連する都市または地区        |
| cb_num       | 411                                 | int     | コミュニティボード番号                   |
| st_senate    | 11                                  | int     | 州上院地区番号                         |
| st_assem     | 25                                  | int     | 州議会地区番号                         |
| cncldist     | 23                                  | int     | 市議会の地区番号                       |

## 役割重複したカラム
役割の重複するカラムは積極的に統合し、ひとつの特徴量としたほうが良い。
理由としては、多重共線性（マルチコ）による線形モデルが使えなくなる懸念と決定木モデルで重要度算出をする場合に不利となることがある。

役割の重複するカラムを以下に示す。
・spc_latinとspc_common
・ntaとnta_name


## ユニーク数の多いカラム
SIGNATE-StudentCupで社会人部門1位の人によると、
ユニーク数が多いカラムはターゲットエンコーディングが有効らしい。

ユニーク数の多いカラムを以下に示す。
| Column      | N train | N test |
|-------------|---------|--------|
| spc_common  | 120     | 118    |
| boro_ct     | 1193    | 971    |
| zip_city    | 45      | 45     |
| cb_num      | 59      | 59     |
| st_senate   | 26      | 26     |
| st_assem    | 65      | 65     |
| cncldist    | 51      | 51     |
| nta, nta_name | 187   | 184    |

## 欠損値について
欠損値は以下の3カラムに存在する。
|  Column    |  NaN count  |
|------------|-------------|
|  steward   |   14883     |
|  guards    |   14943     |
|  problems  |   12243     |

ただし、problemsについてはそもそも問題がない可能性が高く欠損＝問題なしということも考えられる。
そのため、NaNを0としても問題ないだろう。
系統的な欠損値を特徴量としたいのであれば、特定の値の場合に欠損があることを探せば良いかもしれない。
（例：特定の地域で欠損が起こっている、特定のuser_typeで欠損が起こっているなど）

# 木の胴高直径について
ヒストグラムで分布を確認すると以下のような指数分布であった。
![image](https://github.com/rakawanegan/smbc_green_challenge/assets/79976928/e1806bdb-6cb4-4094-b5a5-e59eb798facb)

LightGBMに説明変数の分布の制約はないが、値が大きいものの大小関係を曖昧にし、相対的に小さい大小関係を比較しやすいように自然対数を用いて対数変換してあげると正規分布に近い分布が得られた。
![image](https://github.com/rakawanegan/smbc_green_challenge/assets/79976928/76bf2225-1da5-44fc-9e2a-211536329421)

それにしても明確に多峰的である。
おそらく木の種類によるものだろうと仮定してspcのUniqueごとにヒストグラムをプロットしたが、多峰的であった。
![image](https://github.com/rakawanegan/smbc_green_challenge/assets/79976928/8e312dd9-e785-4edc-913d-59458e8bea13)
![image](https://github.com/rakawanegan/smbc_green_challenge/assets/79976928/b322fd11-8853-432f-8a47-33b074a769e0)

次に考えられるのは地域だろうか。

# 地域情報について

| カラム名          | 説明                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------- |
| boro_ct (4152901)  | ニューヨーク市のセンサストラクトのコード。センサストラクトは市内の地域をさらに細かく区分けした地理的なエリアを指します。   |
| nta (QN45)         | 地域（Neighborhood Tabulation Area）の略称。地域を一意に識別するためのコードや略称が含まれます。                   |
| nta_name           | 地域（Neighborhood Tabulation Area）の正式な名称。地域の名前が含まれます。                              |
| zip_city (Little Neck) | 郵便番号に関連する都市または地区の名前。この場合、Little Neck（リトルネック）が含まれます。                |
| cb_num (411)       | コミュニティボード番号。市内の地域を代表するコミュニティボードの番号が含まれます。                           |
| cncldist (23)      | 市議会の地区番号。ニューヨーク市内の市議会地区を一意に識別するための番号が含まれます。                        |
| boroname (Queens)  | ニューヨーク市の行政区分（ボロー）の名称。例えば、Queens（クイーンズ）などが含まれます。                       |
| borocode (4)       | ニューヨーク市の行政区分（ボロー）のコード。各行政区分を一意に識別するための数字が含まれます。                  |
| st_senate (11)     | 州上院地区番号。ニューヨーク州内の上院地区を一意に識別するための番号が含まれます。                           |
| st_assem (25)      | 州議会地区番号。ニューヨーク州内の議会地区を一意に識別するための番号が含まれます。                         |


GPT君曰く情報の粒度順にするとこんな感じらしいですね。
欠損もないことですしこれならboro_ct だけで情報的には完結していそうですけどどういう使い方をすれば良いんでしょうか

と思ってたらお叱りを受けました。

いいえ、boro_ctだけでは他のすべての情報を計算することはできません。boro_ctはセンサストラクトのコードであり、地理的なエリアを識別するものですが、他の情報は地域や行政区分、コミュニティの詳細な属性を提供しています。
たとえば、ntaやnta_nameは地域の略称や正式な名称を提供し、cb_numはコミュニティボード番号、cncldistは市議会の地区番号、boronameやborocodeは行政区分（ボロー）に関する情報を提供しています。これらの情報はboro_ct単独ではわからない詳細な属性であり、それぞれ異なる側面や粒度の情報を表しています。
したがって、他の情報を計算するには、それぞれのカラムに対応するデータが必要です。


# F1スコアの特性

今回の評価指標はマクロF1スコアであり、各ラベルについてF1スコアを計算しその平均をとる。
F1スコアはRecallとPrecisionの調和平均である。

平均をとるという操作の都合上、外れ値に厳しい。
ここでいう外れ値とは、あるラベルにおいて著しくF1スコアが低いという意味である。

F1スコアはラベルの数が少ないと1予測の誤差が大きく影響してしまう。
よって今回のデータの不均衡性を考慮するのであれば、
特にラベル2のデータについてF1スコアが悪ければそのスコアに引っ張られ、マクロF1スコアが大きく低下してしまう。

対策としては閾値選択について、ラベルの総数が小さい2,0,1の順番で大きく予測を取ったほうが良いだろう。
しかし、擬陽性を大きく検出してしまうとPrecisionが低下してしまうのでおおげさでない適切な閾値が重要である。
まとめると、少数データを外さないようにすることが重要である。


