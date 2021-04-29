import utils
import numpy as np
import random
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

""" Dans ce projet, parfois on utilise objet au lieu de df"""

def getPrior(objet ,argument='target'):
    """
    En calculant le moyen et la variance, on obtient l'intevalle de confiance en 95%

    :objet: dataframe choisit
    :argument: sur  lequel on obtient les informations
    :return: l'information sur ce argument dans ce dataframe
    """
    moy=np.mean(objet[argument])
    var=np.var(objet[argument])
    elemet=objet[argument]
    if isinstance(elemet, int)==True:
        n=1
    else:
        n=len(objet[argument])
    racine_n=(n)**0.5
    racine_var=var**0.5
    std_sur_racine_n=racine_var/racine_n
    min5pourcent=moy-1.96*std_sur_racine_n
    max5pourcent=moy+1.96*std_sur_racine_n
    result={'estimation' :moy, 'min5pourcent':min5pourcent,'max5pourcent': max5pourcent}
    return result

def P2D_l(df, attr):
    """
    Création d'un dictionnaire de la probabilité P (attr|target) pour df
    :df: dataframe choisit
    :attr: argument sur lequel on etudie
    :return un dictionnaire de la probabilité P (attr|target)
    """
    dic_p={1:{}, 0:{}}
    vrai=0
    faux=0
    for t in df.itertuples():
        dic=t._asdict()
        if dic[attr] not in dic_p[dic['target']]:
            dic_p[dic['target']][dic[attr]]=0
        if dic['target']==1:
            vrai+=1
        else:
            faux+=1
    for t in df.itertuples():
        dic=t._asdict()
        dic_p[dic['target']][dic[attr]]+=1
    for i in dic_p:
        for j in dic_p[i]:
            if i==1:
                (dic_p[i])[j]=(dic_p[i])[j]/vrai
            else:
                (dic_p[i])[j]=(dic_p[i])[j]/faux
    return dic_p

def P2D_p(df, attr):
    """
    Création d'un dictionnaire de la probabilité P (target|attr) pour df
    :df: dataframe choisit
    :attr: argument sur lequel on etudie
    :return: un dictionnaire de la probabilité P (target|attr)
    """
    dic_p={}
    for t in df.itertuples():
        dic=t._asdict()
        if dic[attr] not in dic_p:
            dic_p[dic[attr]]={1:0, 0:0}
    for t in df.itertuples():
        dic=t._asdict()
        dic_p[dic[attr]][dic['target']]+=1
    for i in dic_p:
        somme=((dic_p[i])[0]+(dic_p[i])[1])
        for j in dic_p[i]:
                (dic_p[i])[j]=(dic_p[i])[j]/somme
    return dic_p

def PND_l(df, attr=1):
    """
    Création d'un dictionnaire de la probabilité P (attr|target) pour df
    :df: dataframe choisit
    :attr: argument sur lesquels on etudie, on prend tous les arguments par défaut
    """
    if attr==1:
        attr=list(df.columns.values)
    arg=attr[0]

    total={1:{},0:{}}
    vrai=0
    faux=0
    bool=1
    while attr!=[]:
        arg=attr[0]
        if arg=='target':
            attr.remove(arg)
            if attr!=[]:
                arg=attr[0]
            else:
                break
        dic_p={1:{arg:{}}, 0:{arg:{}}}
        for t in df.itertuples():
            dic=t._asdict()
            if dic[arg] not in dic_p[dic['target']][arg]:
                dic_p[dic['target']][arg][dic[arg]]=1
                if dic['target']==1:
                    val=0
                else:
                    val=1
                if dic[arg] not in dic_p[val][arg]:
                    dic_p[val][arg][dic[arg]]=0
            else:
                dic_p[dic['target']][arg][dic[arg]]+=1
            if dic['target']==1 and bool==1:
                vrai+=1
            if dic['target']==0 and bool==1:
                faux+=1
        attr.remove(arg)
        total={1:{**total[1], **dic_p[1]},0:{**total[0], **dic_p[0]}}
        bool=0

    for i in total:
        for j in total[i]:
            for k in total[i][j]:
                if i==1:
                    total[i][j][k]=total[i][j][k]/vrai
                else:
                    total[i][j][k]=total[i][j][k]/faux
    return total

def PND_p(df, attr=1):
    """
    Création d'un dictionnaire de la probabilité P (target|attr) pour df
    :df: dataframe choisit
    :attr: argument sur lesquels on etudie, on prend tous les arguments par défaut
    :return: un dictionnaire de la probabilité P (target|attr)
    """
    if attr==1:
        attr=list(df.columns.values)
    dic_p=dict()
    arg=attr[0]
    if arg=='target':
        attr.remove(arg)
        if attr!=[]:
            arg=attr[0]
        else:
            return {}
    for t in df.itertuples():
        dic=t._asdict()
        if dic[arg] not in dic_p:
            dic_p[dic[arg]]={1:0, 0:0}
    for t in df.itertuples():
        dic=t._asdict()
        dic_p[dic[arg]][dic['target']]+=1
    for i in dic_p:
        somme=((dic_p[i])[0]+(dic_p[i])[1])
        for j in dic_p[i]:
                (dic_p[i])[j]=(dic_p[i])[j]/somme
    attr.remove(arg)
    if attr==[]:
        return {arg:dic_p}
    return {**{arg:dic_p}, **PND_p(df,attr)}


def nbParams(df, liste=None):
    """
    calcul a taille mémoire de ces tables P(target|attr1,..,attrk)
    :df: dataframe choisit
    :liste: la liste des argument
    :pas de return
    """
    if liste==None:
      liste = list(df.columns.values)
    lg=len(liste)
    cpt=len(liste)
    dic=dict()
    result=1

    while(cpt>0):
        dic[lg-cpt]=[]
        for t in df.itertuples():
            d=t._asdict()
            arg=liste[lg-cpt]
            if d[arg] not in dic[lg-cpt]:
                dic[lg-cpt].append(d[arg])
        cpt-=1
    while(lg>0):
       result*=len(dic[lg-1])
       lg-=1
    print(len(liste)," variable(s) : ", 8*result, "octets")
    #return 8*result

def nbParamsIndep(df, liste=None):
    """
    calcul a taille mémoire de ces tables P(target|attr1,..,attrk)
    en supposant l'indépendance des variables.
    :df: dataframe choisit
    :liste: la liste des argument
    :pas de return
    """
    if liste==None:
      liste = list(df.columns.values)
    lg=len(liste)
    cpt=len(liste)
    dic=dict()
    result=0

    while(cpt>0):
        dic[lg-cpt]=[]
        for t in df.itertuples():
            d=t._asdict()
            arg=liste[lg-cpt]
            if d[arg] not in dic[lg-cpt]:
                dic[lg-cpt].append(d[arg])
        cpt-=1
    while(lg>0):
       result+=len(dic[lg-1])
       lg-=1
    print(len(liste)," variable(s) : ", 8*result, "octets")
    #return 8*result

def drawNaiveBayes(df, argument='target'):
    """
    Dessiner un graphe retourne un appel à utils.drawGraph
    :df: dataframe choisit
    :argument: l'argument choisit a etudier
    :appel à utils.drawGraph pour tracer le graohe
    """
    liste = list(df.columns.values)
    string=""
    for classe in liste:
        if classe!=argument:
            string+=argument+"->"+classe+";"
    return utils.drawGraph(string)

def nbParamsNaiveBayes(df, argument, liste=None):
    """
    Calcule la taille mémoire nécessaire pour représenter les tables de probabilité
    en utilisant l'hypothèse du Naive Bayes
    :df: dataframe choisit
    :argument: l'argument choisit a etudier
    :list: une liste des arguments
    :returener la taille mémoire nécessaire
    """
    if liste==None:
        liste = list(df.columns.values)
    lg=len(liste)
    if lg==0:
        print(len(liste)," variable(s) : ", 16, "octets")
        return 16
        #return 16
    cpt=len(liste)
    dic=dict()
    result=0
    while(cpt>0):
        dic[lg-cpt]=[]
        for t in df.itertuples():
            d=t._asdict()
            arg=liste[lg-cpt]
            if d[arg] not in dic[lg-cpt]:
                dic[lg-cpt].append(d[arg])
        cpt-=1
    while(lg>0):
        result+=len(dic[lg-1])
        lg-=1
    if "target" in liste:
        result-=2
    result=result*2+2
    print(len(liste)," variable(s) : ", 8*result, "octets")
    return result



def isIndepFromTarget(df, attr, x):
    """
    Determiner si un attr est independant de target
    :df: dataframe choisit
    :attr: l'argument choisit a etudier
    :x: seuil
    :retourner true si n attr est independant de target
    """
    obs=[[], []]
    ref=[]
    for t in df.itertuples():
        dic=t._asdict()
        if dic[attr] not in ref:
            ref.append(dic[attr])
        index=ref.index(dic[attr])
        if(len(obs[0])<index+1):
            obs[0].append(0)
            obs[1].append(0)
        obs[dic['target']][index]+=1
    a,b,c,d=chi2_contingency(obs)
    if b<x:
        return False
    return True
class APrioriClassifier(utils.AbstractClassifier):

    def __init__(self):
        super().__init__()
        self.type=1
        self.objet=None
        self.argument='target'
        self.t_P2DL={}
        self.t_P2DP={}
        self.val=0



    def estimClass(self, objet):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param objet: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        if objet is None:
            return
        result=getPrior(objet)
        if self.type==1:
            if(result['estimation']>0.5):
                return 1
            return 0
        if self.type>=4:
            r=self.estimProbas(objet)
            if r[0]>=r[1]:
                return 0
            return 1
        dic=dict()
        t_P2DL=self.t_P2DL
        t_P2DP=self.t_P2DP
        result1=objet[self.argument]
        if t_P2DL!={}:
             for t in t_P2DL:
                 for k in t_P2DL[t]:
                     if k not in dic:
                         dic[k]=[-1,-1]
                     if dic[k][1]<t_P2DL[t][k]:
                         dic[k][1]=t_P2DL[t][k]
                         dic[k][0]=t
             return dic[result1][0]

        if t_P2DP!={}:
             for t in t_P2DP:
                 if t not in dic:
                     dic[t]=[-1,-1]
                 for k in t_P2DP[t]:
                     if dic[t][1]<t_P2DP[t][k]:
                         dic[t][1]=t_P2DP[t][k]
                         dic[t][0]=k
             return dic[result1][0]


    def estimProbas(self,  df=None):
        """
        à partir d'un dictionanire d'attributs, retourner p de la classe 0 ou 1
        :df: dataframe choisit
        :return: dictionnaire de possibilité
        """
        if self.type==4:
            table=self.t_P2DL
        else:
            table=self.t_P2DL

        result1=1
        result2=1
        liste = self.argument
        if 'target' in liste:
            liste.remove('target')
        if len(table)==2 and (self.type==4 or self.type==6):
            for t in liste:
                if  df[t] not in table[0][t]:
                    result1=0
                else:
                    result1=result1*table[0][t][df[t]]
                if df[t] not in table[1][t]:
                    result2=0
                else:
                    result2=result2*table[1][t][df[t]]
            return {0:result1, 1:result2}
        else:
            result1=1
            result2=1
            if (self.type==5):
                new=MLNaiveBayesClassifier(self.objet)
            else:
                new=ReducedMLNaiveBayesClassifier(self.objet, self.val)
            estim=new.estimProbas(df)

            result1=estim[0]
            result2=estim[1]
            vrai=0
            somme=0
            for t in self.objet.itertuples():
                 dic=t._asdict()
                 somme+=1
                 if dic['target']==1:
                     vrai+=1
            p=vrai/somme
            s=result1*(1-p)+(result2)*p
            if s!=0:
                return { 0:result1*(1-p)/s, 1:(result2)*p/s}
            return {0:0, 1:0}



    def statsOnDF(self, objet):
        """
        Une méthode statsOnDF qui rendra les valeurs VP, VN, FP et FN
        :objet:dataframe choisit
        :return: VP, VN, FP et FN
        """
        VP=VN=FP=FN=0
        for t in objet.itertuples():
             dic=t._asdict()
             if self.type==1:
                 classe_prevu=self.estimClass(objet)
             else:
                 classe_prevu=self.estimClass(dic)
             if(classe_prevu==1):
                 if dic['target']==1:
                     VP+=1
                 else:
                     FP+=1
             else:
                if dic['target']==1:
                    FN+=1
                else:
                    VN+=1
        if (VP+FP)==0:
            result1=0
        else:
            result1=VP/(VP+FP)
        if (VP+FN)==0:
            result2=0
        else:
            result2=VP/(VP+FN)
        return {'VP' : VP, 'VN' : VN, 'FP' :FP, 'FN':FN, 'Precision' : result1, 'Rappel':result2}

class ML2DClassifier(APrioriClassifier):

    def __init__(self, objet, argument):
        super().__init__()
        self.type=2
        self.objet=objet
        self.argument=argument
        self.t_P2DL=P2D_l(objet, argument)
        self.t_P2DP={}

class MAP2DClassifier(APrioriClassifier):

    def __init__(self, objet, argument):
        super().__init__()
        self.type=3
        self.objet=objet
        self.argument=argument
        self.t_P2DL={}
        self.t_P2DP=P2D_p(objet, argument)

class MLNaiveBayesClassifier(APrioriClassifier):
    def __init__(self, objet):
        super().__init__()
        self.type=4
        self.objet=objet
        self.argument=list(objet.columns.values)
        self.t_P2DL=PND_l(objet)
        self.t_P2DP={}

class MAPNaiveBayesClassifier(APrioriClassifier):
    def __init__(self, objet):
        super().__init__()
        self.type=5
        self.objet=objet
        self.argument=list(objet.columns.values)
        self.t_P2DL={}
        self.t_P2DP=PND_p(objet)

class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    def __init__(self, objet, p):
        super().__init__(objet)
        self.type=6
        self.argument=[]
        liste=list(objet.columns.values)
        for a in liste:
            if(isIndepFromTarget(objet, a, p)==0) and a!='target':
                self.argument.append(a)
        self.t_P2DL=PND_l(objet)
        self.t_P2DP={}
        self.val=p

    def draw(self, argument='target'):
        """
        a partir d'un dataframe et du nom de la colonne qui est la classe, dessine le graphe.
        :argument: l'argument choisit
        :return: l'image représentant le graphe
        """
        liste = self.argument
        string=""
        for classe in liste:
            if classe!=argument:
                string+=argument+"->"+classe+";"
        return utils.drawGraph(string)

class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    def __init__(self, objet, p):
        super().__init__(objet)
        self.type=7
        self.argument=[]
        liste=list(objet.columns.values)
        for a in liste:
            if(isIndepFromTarget(objet, a, p)==0) and a!='target':
                self.argument.append(a)
        self.t_P2DL={}
        self.t_P2DP=PND_p(objet, self.argument)
        self.val=p

    def draw(self, argument='target'):
        """
        a partir d'un dataframe et du nom de la colonne qui est la classe, dessine le graphe.
        :argument: l'argument choisit
        :return: l'image représentant le graphe
        """
        liste = self.argument
        string=""
        for classe in liste:
            if classe!=argument:
                string+=argument+"->"+classe+";"
        return utils.drawGraph(string)

class mapClassifiers:
    def __init__(self, dic, objet):
        self.dic=dic
        a=dict()
        print("Veuillez patienter car l'éxécution de classe 4 et 6 prend du temps")
        for t in dic:
            print("Debut de ", t)
            new=dic[t]
            val=new.statsOnDF(objet)
            a[t]={'Precision':val['Precision'],'Rappel': val['Rappel']}
            print("Fin de ", t)
        print(a[t])
        for t in a:
            print("Info sur ", t, ":", a[t]['Rappel'], a[t]['Precision'])
            plt.plot(a[t]['Precision'],a[t]['Rappel'], color='red', linestyle='dashed', linewidth = 3,
         marker='x', markersize=5)
            plt.annotate(t, (a[t]['Precision'],a[t]['Rappel']))
