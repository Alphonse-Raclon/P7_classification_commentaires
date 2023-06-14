import pandas as pd
import nltk
import matplotlib.pyplot as plt


class CommentClassification:
    """docstring for CommentClassification"""

    def __init__(self):
        self.__common_keys = None
        self.__df_freq_pond_pos = None
        self.__df_freq_pond_neg = None

    def __frequence_mot(self, texte_serie):
        """
        avis_serie_n : Série pandas contenant des listes de mots (un mot pouvant en regrouper un ou plus)

        return : DataFrame avec une colonne "frequence" contenant la fréquence de chaque mot rangé par ordre décroissant
        """
        liste_mots_complete = []
        for texte in texte_serie:
            liste_mots_complete += texte

        freq = nltk.FreqDist(liste_mots_complete)

        for mot, qtt in freq.items():
            freq[mot] = [qtt]
        df_freq = pd.DataFrame(freq).transpose().loc[:, 0].sort_values(ascending=False)
        df_freq = pd.DataFrame(df_freq)
        df_freq.columns = ['frequence']

        return df_freq

    def fit(self, neg_comm, pos_comm, freq_min=None):
        """
        :param neg_comm: Series pandas contenant des listes de mots issus de commentaires negatifs
        et normalises de preference
        :param pos_comm: Series pandas contenant des listes de mots issus de commentaires positifs
        et normalises de preference
        :param freq_min: Int frequence en dessous de laquelle on ne gardera pas les mots.
        A ne preciser que si on cherche a optimiser le modele
        """

        # Frequence des commentaires positifs
        df_freq_neg = self.__frequence_mot(neg_comm)
        # Frequence des commentaires negatifs
        df_freq_pos = self.__frequence_mot(pos_comm)

        if freq_min is None:
            freq_min = (df_freq_neg[0] - df_freq_neg[-1]) ** 0.5

        freq_min = min(freq_min * len(neg_comm), freq_min * len(pos_comm))
        df_freq_neg *= len(pos_comm)
        df_freq_pos *= len(neg_comm)

        df_freq_neg = df_freq_neg[df_freq_neg.frequence > freq_min]
        df_freq_pos = df_freq_pos[df_freq_pos.frequence > freq_min]

        df_freq_neg = df_freq_neg['frequence'].to_dict()
        df_freq_pos = df_freq_pos['frequence'].to_dict()

        df_freq_neg = {key: df_freq_neg.get(key, 0.75 * freq_min) for key in set(df_freq_neg) | set(df_freq_pos)}
        df_freq_pos = {key: df_freq_pos.get(key, 0.75 * freq_min) for key in set(df_freq_neg) | set(df_freq_pos)}

        common_keys = set(df_freq_neg.keys()) & set(df_freq_pos.keys())
        df_freq_pond_neg = dict(zip(common_keys, map(lambda k: df_freq_neg[k] / df_freq_pos[k], common_keys)))
        df_freq_pond_pos = dict(zip(common_keys, map(lambda k: df_freq_pos[k] / df_freq_neg[k], common_keys)))

        df_freq_pond_neg = pd.DataFrame.from_dict(df_freq_pond_neg, orient='index', columns=['frequence'])
        df_freq_pond_neg = df_freq_pond_neg.sort_values(by='frequence', ascending=False)

        df_freq_pond_pos = pd.DataFrame.from_dict(df_freq_pond_pos, orient='index', columns=['frequence'])
        df_freq_pond_pos = df_freq_pond_pos.sort_values(by='frequence', ascending=False)

        self.__df_freq_pond_neg = df_freq_pond_neg
        self.__df_freq_pond_pos = df_freq_pond_pos
        self.__common_keys = common_keys

        print("L'entrainement s'est effectue correctement")

    def histogramme(self, nbr_mots):
        """
        affiche les histogrammes des mots les plus courants pour les classes de commentaires négatifs et positifs.

        :param nbr_mots: entier indiquant le nombre de mots les plus usités à afficher dans l'histogramme.
        """
        df_freq_first50 = self.__df_freq_pond_neg.iloc[:nbr_mots, :]
        df_freq_first50.plot(
            kind='bar', use_index=True, figsize=(15, 5),
            title="Fréquence pondérée décroissante d'apparition des {} premiers mots négatifs".format(nbr_mots))
        plt.gca().tick_params(axis='x', length=0)
        plt.show()

        df_freq_first50 = self.__df_freq_pond_pos.iloc[:nbr_mots, :]
        df_freq_first50.plot(
            kind='bar', use_index=True, figsize=(15, 5),
            title="Fréquence pondérée décroissante d'apparition des {} premiers mots positifs".format(nbr_mots))
        plt.gca().tick_params(axis='x', length=0)
        plt.show()

    def predict_proba_class(self, neg_or_pos_comm):
        """
        :param neg_or_pos_comm: Series pandas contenant des listes de mots issus de commentaires et
         normalises de preference
        """

        neg_or_pos_comm = neg_or_pos_comm.apply(lambda x: nltk.FreqDist(x))
        neg_or_pos_comm = neg_or_pos_comm.apply(lambda dic: {key: dic.get(key, 0) for key in self.__common_keys})

        neg_or_pos_comm = neg_or_pos_comm.apply(
            lambda dic: pd.concat(
                [pd.DataFrame.from_dict(dic, orient='index', columns=['commentaire']),
                 self.__df_freq_pond_neg.rename(columns={'frequence': 'frequence_neg'}).copy(),
                 self.__df_freq_pond_pos.rename(columns={'frequence': 'frequence_pos'}).copy()],
                axis=1
            ).corr().commentaire.loc[["frequence_neg", "frequence_pos"]]
        )

        res = pd.DataFrame(
            {
                "confiance": (0.5 * abs(neg_or_pos_comm.frequence_neg - neg_or_pos_comm.frequence_pos)).fillna(-1),
                "prediction": neg_or_pos_comm.idxmax(axis=1).fillna('neg').apply(lambda x: x[-3:])
            }
        )

        return res


def eta_squared(df, variable, label):
    moyenne_y = df[variable].mean()
    classes = []
    for classe in label.unique():
        yi_classe = df[label == classe]
        classes.append({'ni': yi_classe.shape[0],
                        'moyenne_classe': yi_classe.mean()[0]})
    df = (df - moyenne_y) ** 2
    SCT = df.sum()
    SCE = sum([c['ni'] * (c['moyenne_classe'] - moyenne_y) ** 2 for c in classes])
    return (SCE / SCT)[0]


def anova(df, variable, label):
    eta2 = eta_squared(df, variable, label)
    labels = df[variable].unique()
    n_labels = labels.shape[0]
    box = []
    for i in labels:
        box += [df.loc[df[label] == i, df[variable]]]
    medianprops = {'color': 'navy'}
    meanprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'white'}
    plt.figure(figsize=(12, 8))
    box1 = plt.boxplot(box, labels=labels, vert=True, showmeans=True, showfliers=False, patch_artist=True,
                       medianprops=medianprops, meanprops=meanprops)
    color = ['lightskytblue', 'aquamarine', 'yellow', 'tomato', 'mediumorchid']
    for i in range(n_labels):
        plt.setp(box1["boxes"][i], facecolor=color[i % n_labels])

    plt.title('Repartition de l\'indice de confiance en fonction de la justesse ou non de la prédiction', size=20)
    plt.figtext(0.15, 0.82, '$\eta^2$={}'.format(round(eta2, 3)), fontsize=15)
    plt.ylabel(variable)
    plt.show()
