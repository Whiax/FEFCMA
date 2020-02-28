# -*- coding: utf-8 -*-
from imports import *
from consts import *
from functools import lru_cache, wraps
import numpy as np
# =============================================================================
#  METHODES
# =============================================================================
#
def freezeargs(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([e if type(e) != list else tuple(e) for e in args])
        kwargs = {k: frozenset(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped 


#Cherche les fichiers recursivement à partir d'un dossier
@lru_cache(maxsize=128)
def recursiveFileFind(folder):
    files = []
    for root, dirs, folderfiles in os.walk(folder):
        for file in folderfiles:
            pth = os.path.join(root, file)
            files.append(pth)
    return files

#met en une liste une liste de liste
def linify(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

#compte les caractères via un Counter
@lru_cache(maxsize=128)
def charCount(words_list):
    # Occurences des caractères
    one_line = ''.join(words_list)
    char_count = Counter(one_line)
    char_count = sorted(char_count.items(),  key=lambda x: -x[1])
    return char_count

# mots => caractères
def split(word): 
    return [char for char in word]  

#d => sorted d
def sortd(d):
    return sorted(d.items(),  key=lambda x: -x[1])

#d/listoflist => list
def flattenb(l):
    lb = []
    l = l.values() if type(l) == dict else l
    for e in l:
        if type(e) == list:
            lb.extend(e)
        else:
            lb.append(e)
    return lb


#list of languages
@lru_cache(maxsize=128)
def getLanguages():
    files = recursiveFileFind(source_multilang)
    conllu = [f for f in files if f.endswith('.conllu')]
    languages = list(set([basename(dirname(f)) for f in conllu])) 
    return languages

#charge tous les mots d'une langue
@lru_cache(maxsize=20)
def loadWords(lang, punct=False):
    if lang == 'voynich':
        v101lines = loadv101lines(v101path)
        v101words = loadv101words(v101lines)
        return v101words
    files = recursiveFileFind(source_multilang)
    conllu = [f for f in files if f.endswith('.conllu')]
    lang_files = [file for file in conllu if basename(dirname(file)) == lang]
    lines = [open(file, encoding='utf-8').readlines() for file in lang_files]
    lines = linify(lines)
    lines = [l.replace('\n','') for l in lines]
    words = [l.split('\t')[1] for l in lines if len(l.split('\t')) >= 2]
    
    #prétraitement qui supprime toutes les ponctuations (aide pour comparer avec voynich?)
    words_bis = []
    for w in words:
        b_in = True
        for c in w:
            if c in string.punctuation:
                b_in = False
                break
        if b_in:
            words_bis.append(w)
    words = words_bis
    return words

#compte les mots
@freezeargs
@lru_cache(maxsize=128)
def countWords(words, percentage=False):
    words_count = Counter(words)
    n_word = sum(words_count.values())
    if percentage:
        for k,v in words_count.items():
            words_count[k] = words_count[k]/n_word
    words_count_l = sorted(words_count.items(),  key=lambda x: -x[1])
    words_count = OrderedDict(words_count_l)
    return words_count, words_count_l

#compte les caractères
@freezeargs
@lru_cache(maxsize=128)
def countChars(words, percentage=False):
    one_line = ''.join(words)
    char_count = Counter(one_line)
    n_char = sum(char_count.values())
    if percentage:
        for k,v in char_count.items():
            char_count[k] = char_count[k]/n_char
    char_count_l = sorted(char_count.items(),  key=lambda x: -x[1])
    char_count = OrderedDict(char_count_l)
    return char_count, char_count_l

#plot la loi de zipf
def zipf(words_count,name='Unnamed',save=False):
    print("Loi de Zipf")
    x = list(range(1,len(words_count)+1))
    y = list(words_count.values())
    plt.plot(x,y,'x', label=name)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(name)
    plt.legend()
    if not save:
        plt.show()
    plt.savefig('publications/'+name)
    
#charge les lignes du voynich
@freezeargs
@lru_cache(maxsize=128)
def loadv101lines(path):
    v101lines = open(path).readlines()
    v101lines = [l.replace('\n','') for l in v101lines]
    v101lines = [re.sub(r'^<.*>','',l) for l in v101lines]
    v101lines = [re.sub(r'^<!.*>','',l) for l in v101lines]
    v101lines = [re.sub(r'^#.*','',l) for l in v101lines]
    v101lines = [l.lstrip().rstrip() for l in v101lines]
    v101lines = [l for l in v101lines if l != '']
    return v101lines

#charge les mots du voynich
@freezeargs
@lru_cache(maxsize=128)
def loadv101words(v101lines):
    v101lines = list(v101lines) if type(v101lines) == tuple else v101lines
    #remplacer les @X; par des symboles et ajouter à la liste
    #v101lines = [ l  for l in v101lines if '@' in l]
    shift = 200
    for i,l in enumerate(v101lines):
        l = l.split('.')
        l = [li.split(';') for li in l]
        lss = []
        for ls in l:
            ls = [str(chr(int(li.split('@')[1])+shift)) if '@' in li else li for li in ls]
            lss.append(''.join(ls))
        l = '.'.join(lss)
        v101lines[i] = l
    #v101lines = [int(l) for l in v101lines]l
    
    #supprimer les <$> => A remplacer par une restructuration des lignes par paragraphes
    v101lines = [re.sub(r'<$>','',l) for l in v101lines]
    
    #créer les mots en remplaçant les '.'
    v101lines = [re.sub(r'\.',' ',l) for l in v101lines]
    
    #créer les mots en remplaçant les ','
    v101lines = [re.sub(r',',' ',l) for l in v101lines]
    
    #liste des mots
    v101words = [l.split(' ') for l in v101lines]
    v101words = linify(v101words)
    return v101words

# Reconstruit les clé du dictionnaire des statistiques
def getStatsKeys(stats_dict):
    fields = []
    for i,(k,v) in enumerate(stats_dict[list(stats_dict.keys())[0]].items()):
        fieldname_or = k
        if type(v) == list:
            for sub_i in range(len(v)):
                fieldname_sub = fieldname_or + '_'+str(sub_i)
                if type(v[sub_i]) == list:
                    for sub_sub_i in range(len(v[sub_i])):
                        fieldname_subsub = fieldname_sub + '_'+str(sub_sub_i)
                        fields.append(fieldname_subsub)
                else:
                    fields.append(fieldname_sub)
        else:
            fields.append(fieldname_or)
    return fields

# Normalise les langues
def normalizeLang(data):
    n0 = data.shape[0]
    mean = np.tile(data.mean(0),n0).reshape((n0,-1))
    std = np.tile(data.std(0),n0).reshape((n0,-1))
    mindata = np.tile(data.min(0),n0).reshape((n0,-1))
    maxdata = np.tile(data.max(0),n0).reshape((n0,-1))
    data = (data-mean)/(std)
    return data



# Build stats
def buildStats(lang):
    
    stats = OrderedDict()
    
    # Quantiles/Moyennes des tailles des mots : 0.1, 0.25, 0.5, 0.75, 0.9
    n_mots = len(words)
    if n_mots == 0:
        return None
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    tailles = [len(w) for w in words]
    tailles = np.sort(tailles)
    stats['tailles_quantiles'] = [tailles[int(i*n_mots)] for i in quantiles]
    stats['tailles_moyennes'] = [np.mean(tailles[int(quantiles[i-1]*n_mots):int(quantiles[i+1]*n_mots)]) for i in [1,2,3]]
    stats['tailles_moyennes'].append(np.mean(tailles))
    
    # Variance sur la taille des mots
    stats['tailles_variances'] = np.var(tailles)
    
    # Proportions de caractère unique par mot (aaba : 2/4), variance, moyenne
    splits = [split(w) for w in words]
    pcharupw = np.sort([len(set(sw))/len(sw) if len(sw)>0 else 0 for sw in splits])
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5]
    stats['pcharupw'] = [np.mean(pcharupw),np.var(pcharupw)]
    stats['pcharupw'].append([pcharupw[int(i*n_mots)] for i in quantiles])
    
    # Proportion des N - M-grams les plus fréquents en pourcentage du total des M-grams
    Ns = [1,2,3,5,7,10,15,50,100]
    Ms = [1,2,3,4]
    chars_dict = {m:dict() for m in Ms}
    for w in words:
        for M in Ms:
            if len(w) >= M:
                for i,c in enumerate(w):
                    Mgram = w[i:i+M]
                    if not Mgram in chars_dict[M].keys(): 
                        chars_dict[M][Mgram] = 0
                    chars_dict[M][Mgram] += 1
                    if i+M-1 >= len(w):
                        break
                    
    # Trie (accessoirement) les tableaux générés
    p_ngram = []
    for k,d in chars_dict.items():
        d = sortd(d)
        d = OrderedDict(d)
        chars_dict[k] = d
        
        #cas particulier (précision langues romanes)
        used_Ns = Ns
        if k in [1]:
            used_Ns = Ns[0:5]
        
        # Sommes locales et globales
        s_glob = sum(d.values())
        proportion = [sum(list(d.values())[0:n])/s_glob if s_glob != 0 else 0 for n in used_Ns]
        p_ngram.append(proportion)
    stats['n*mgram'] = p_ngram
    
    # Proportion des N mots les plus fréquents en pourcentage du total des mots 
    # MALCOMPATIBLE PETIT CORPUS
    #N = [1,10,100,1000]
    #n_words = len(words)
    #words_count = OrderedDict(sortd(Counter(words)))
    #words_count_values = list(words_count.values())
    #p_frequent_words = [sum(words_count_values[0:n])/n_words for n in N]
    #stats['pfwords'] = p_frequent_words
    
    # Nombre de mots uniques en pourcentage du nombre de mot en moyenne pour M mots random sur N iteration
    N = 20
    M = 500
    rint = np.random.randint(0,len(words),(N, M))
    packs_1000w = [[words[i] for i in l] for l in rint]
    stats['prop_uniq_words_m'] = np.mean([len(set(ws))/len(ws) for ws in packs_1000w])
    stats['prop_uniq_words_v'] = np.var([len(set(ws))/len(ws) for ws in packs_1000w])
    
    # Pour chaque mot, en fonction de sa taille, taille moyenne du mot suivant & précédent
    tailles = [1,2,3,4,5]
    prec_next = [-1,1]
    pos_size_rel = {t:list() for t in tailles}
    pos_size_abs = {t:list() for t in tailles}
    for pos in prec_next:
        for i,w in enumerate(words):
            if i+pos >= 0 and i+pos < len(words) and len(w) in tailles:
                pos_size_rel[len(w)].append(len(words[i+pos])/len(w))
                pos_size_abs[len(w)].append(len(words[i+pos]))
    for k in tailles:
        pos_size_abs[k] = np.mean(pos_size_abs[k]) if len(pos_size_abs[k]) > 0 else 0
        pos_size_rel[k] = np.mean(pos_size_rel[k]) if len(pos_size_rel[k]) > 0 else 0
    stats['pos_size_abs'] = list(pos_size_abs.values())
    stats['pos_size_rel'] = list(pos_size_rel.values())
    
    # Proportions de M-gram uniques en début/fin mot
    # TODO
    # Ms = [1,2,3]
    
    # Proportion des mots contenant une ponctuation
    stats['prop_punct'] = np.mean([c in string.punctuation for w in words for c in w])
    
    
    # Proportion des mot de N (1,2,3) caractère
    Ns_prop = [1,2,3,4,5,6]
    props = {k:0 for k in Ns_prop}
    for w in words:
        if len(w) in Ns_prop:
            props[len(w)] += 1
    props = {k:v/len(words) for k,v in props.items()}
    stats['props_len'] = list(props.values())
    return stats


#column ponderation
def columnPonderation(data, amplitude, best_weights=[]):
    weights = np.random.uniform(0,10000,data.shape[1])
    if np.random.uniform(0,1) < 0.8 and len(best_weights)>0:
        weights = best_weights + np.random.uniform(-0.5,0.5,data.shape[1]) * best_weights
            
    return data*weights, weights

#evaluation for langs
def evaluateLangScore(data_opt, languages):
    #evalue
    score = 0
    #pour chaque ligne
    for i,li in enumerate(data_opt):
        lang = languages[i]
        lang = lang.split('UD_')[1].split('-')[0] if 'UD' in lang else lang
        distances = [np.sum(np.abs(li-lj)) if i != j else np.inf for j,lj in enumerate(data_opt) ]
        firsts = np.argsort(distances)[0:5]
        closest_langs = [languages[l_id] for l_id in firsts]
        closest_langs = [c.split('UD_')[1].split('-')[0] if 'UD' in c else c for c in closest_langs]
        score += np.sum([lang in c for c in closest_langs])
    return score

#column ponderation to optimize proximity between languages (could be based on meta-representation w/ upos & co)
def optimizeData(data, opt_niter, best_score, best_weights, evaluation_method, **kwargs):
    amplitude = kwargs['amplitude'] if 'amplitude' in kwargs else 10000
    for i_iter in range(int(opt_niter)):
        #génère le postprocess
        data_opt, weights = columnPonderation(data, amplitude, best_weights)
        
        #evalue
        score = evaluation_method(data_opt, **kwargs)
        
        
        if score > best_score:
            best_score = score
            best_weights = weights
            print(best_score)
    return best_score, best_weights 

#Sauvegarde data pour projection sur tensorflow projector
def saveData(data, languages):
    data_tsv = np.array(data, dtype=str)
    data_tsv = ['\t'.join(d) for d in data_tsv]
    data_tsv = '\n'.join(data_tsv)
    open("data/lang_embeddings.tsv", "w").write(data_tsv)
    open("data/lang_embeddings_names.tsv", "w").write('\n'.join(languages))



def basicNumStats(val_list, quantiles):
    l_sorted = np.sort(val_list)
    q = [l_sorted[int(i*len(l_sorted))] for i in quantiles if i != 1]
    m = [np.mean(val_list)]
    return linify([q,m])



def getCharsStats(lang):
    data = {}
    #Chargement des mots 
    words = loadWords(lang, punct=False) 
    charscount,cc = countChars(words, percentage=True)
    chars = list(charscount.keys())
    
    # - position en pourcentage des positions totales et des mots totaux 
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    ranges = [0] + quantiles + [1]
    cstats = {c:{'position_by_words':[],'position_by_uwords':[], 'word_avg_size':[],\
                 'avg_positions':[], 'frequency':[]} for c in chars}
    for w in words:
        for i,c in enumerate(list(w)):
            cstats[c]['position_by_words'].append(i/len(w))
            cstats[c]['avg_positions'].append(i/len(w))
            cstats[c]['word_avg_size'].append(len(w))
    for c in chars:
        cstats[c]['position_by_words'] = basicNumStats(cstats[c]['position_by_words'], quantiles)
        cstats[c]['word_avg_size'] = basicNumStats(cstats[c]['word_avg_size'], quantiles)
    alls_dict = {}
    for q1,q2 in zip(ranges[0:len(ranges)-1],ranges[1:]):
        alls_dict[(q1,q2)] = [v for ci in chars for v in cstats[ci]['avg_positions'] if v >= q1 and v < q2]
    pos_list = {}
    for c in chars:
        l = []
        for q1,q2 in zip(ranges[0:len(ranges)-1],ranges[1:]):
            vals = [v for v in cstats[c]['avg_positions'] if v >= q1 and v < q2]
            l += [len(vals)/len(alls_dict[(q1,q2)]) if len(vals) > 0 else 0]
        pos_list[c] = l
    for c in chars:
        cstats[c]['avg_positions'] = pos_list[c]
        
    #, en pourcentage des mots uniques 
    for w in set(words):
        for i,c in enumerate(list(w)):
            cstats[c]['position_by_uwords'].append(i/len(w))
    for c in chars:
        cstats[c]['position_by_uwords'] = basicNumStats(cstats[c]['position_by_uwords'], quantiles)
    
    # - fréquence du caractère
    for k,v in cstats.items():
        cstats[k]['frequency'] = charscount[k]
    
    
    data = np.array([flattenb(e) for e in list(cstats.values())])
    #data = normalizeLang(data)
    data = {'chars':chars,'data':data}
    return data





#Renvoie la proximité entre deux langues (basé sur la substitution de caractères)
def getProximity(langA, langB, mode='random_rank', **options):
    #Lis la config
    is_random = mode=='random_rank'
    is_one_to_one = mode=='one-to-one'
    is_custom = type(mode) == dict
    assert is_random or is_one_to_one or is_custom
    
    # Mapping à calculer
    #Chargement des mots des deux langues
    wordsA = loadWords(langA, punct=options['punct']) 
    wordsB = loadWords(langB, punct=options['punct'])
    
    wordscountB,wcB = countWords(wordsB, percentage=True)
    if not is_custom:
        
        #On ne mappe que les N caractères les plus fréquents
        #N = options['N_char']
        
        #En se basant sur la fréquence des caractères
        #mapC = np.zeros((N,N))
        charscountA,ccA = countChars(wordsA, percentage=True)
        charscountB,ccB = countChars(wordsB, percentage=True)
        keysA = list(charscountA.keys())
        keysB = list(charscountB.keys())
        
        #Mapping
        char_map = {}
        max_rank = options['max_rank']
        possibilities = [i for i in range(len(keysB))]
        for i,kA in enumerate(keysA):
            #map du char avec son identique
            if is_one_to_one:
                char_map[kA] = kA
            #map du char random avec distance en rang
            if is_random:
                c_possible = [j for j in possibilities if np.abs(i-j) < max_rank]
                c_target = random.choice(c_possible) if len(c_possible) > 0 else len(keysB)
                char_map[kA] = keysB[c_target] if c_target < len(keysB) else '_'
                
                #supprime le caractère choisi des possibilités
                if c_target in possibilities:
                    possibilities.remove(c_target)
    # Mapping custom
    else:
        char_map = mode

    #Appliquer le mapping
    new_words = []
    n_words = 50000
    wordsA_short = wordsA[0:n_words]
    for w in wordsA_short:
        new_word = ''
        for i in range(len(w)):
            c = w[i]
            new_c = char_map[c] if c in char_map else '_'
            new_word += new_c
        new_words.append(new_word)
    wordscountNew,wcN = countWords(new_words, percentage=True)
    
    #Mesurer le score par la proportion de nouveaux mots existants dans l'ancien dictionnaire
    #en pourcentage pondéré par l'occurence
    corpus_coverage = 0
    unique_word_coverage = 0
    found_words = []
    for w1 in wordscountNew.keys():
        #distance_valide = 1 if w1 in wordscountNew else 0
        #Si le mot créé existe
        if w1 in wordscountB:
#            print(w1,wordscountNew[w1] )
            found_words.append((w1,wordscountNew[w1]))
            corpus_coverage += wordscountNew[w1] # je regarde quelle couverture j'ai en voynich généré
            unique_word_coverage += 1
            
            #TODO TODO TODO : Compter +1 pour tous les mots à une distance de 1 (doit être intelligent, taille>2)
#        else:
#            print(w1)
    #print('\n'.join([str(s) for s in found_words[0:20]]))
    unique_word_coverage /= len(wordscountNew)
    
    corpus_coverage = np.round(corpus_coverage*100,2)
    unique_word_coverage = np.round(unique_word_coverage*100,2)
    
    return {'corpus_coverage':corpus_coverage, 'unique_word_coverage':unique_word_coverage}


def randomMapping(charsA, charsB):
    #Shortest mapping
    cmap = dict()
    charsA_b = charsA.copy()
    charsB_b = charsB.copy()
    random.shuffle(charsA_b)
    random.shuffle(charsB_b)
    for cA,cB in zip(charsA_b, charsB_b):
        cmap[cA] = cB
    return cmap

def shortestMapping(charsA, charsB, distances):
    #Shortest mapping
    cmap = dict()
    for y,k in enumerate(charsA):
        x = distances[y,:].argmin()
        cmap[k] = charsB[x] 
        distances[:,x] = np.inf
    return cmap

def heuristicShortestMapping(charsA, charsB, distances, a=None, p=None):
    #Shortest mapping
    cmap = dict()
    amplitude = 2 if not a else a
    for y,k in enumerate(charsA):
        x = random.choice(np.argsort(distances[y,:])[0:amplitude])
        if p:
            x = random.choice(np.argsort(distances[y,:])[0:5])
        cmap[k] = charsB[x] 
        distances[:,x] = np.inf
    return cmap

def bruteforceAllMapping(charsA, charsB, distances):
    #Shortest mapping
    cmap = dict()
    distances_copy = distances.copy()
    total_cost = 0
    priciest_mapping = None
    priciest_mapping_score = 0
    for y,k in enumerate(charsA):
        x = distances[y,:].argmin()
        cmap[k] = charsB[x] 
        distances_copy[:,x] = np.inf
        cost = distances[y,x]
        if cost > priciest_mapping_score and cost != np.inf:
            priciest_mapping_score = cost
            priciest_mapping = (y,x)
        total_cost += distances[y,x]
    #print(priciest_mapping, priciest_mapping_score)
    distances[priciest_mapping] = np.inf
    return cmap, total_cost



def transcript(cmap, lang):
    words = loadWords(lang, punct=False) 
        
    #Appliquer le mapping
    new_words = []
    n_words = 50000
    for w in words:
        new_word = ''
        for i in range(len(w)):
            c = w[i]
            new_c = cmap[c] if c in cmap else '_'
            new_word += new_c
        new_words.append(new_word)
        
    return new_words



def foundWords(langA, langB, char_map, **options):
    
    # Mapping à calculer
    #Chargement des mots des deux langues
    wordsA = loadWords(langA, punct=options['punct']) 
    wordsB = loadWords(langB, punct=options['punct'])
    
    wordscountB,wcB = countWords(wordsB, percentage=True)

    #Appliquer le mapping
    new_words = []
    n_words = 50000
    wordsA_short = wordsA[0:n_words]
    for w in wordsA_short:
        new_word = ''
        for i in range(len(w)):
            c = w[i]
            new_c = char_map[c] if c in char_map else '_'
            new_word += new_c
        new_words.append(new_word)
    wordscountNew,wcN = countWords(new_words, percentage=True)
    
    #Mesurer le score par la proportion de nouveaux mots existants dans l'ancien dictionnaire
    #en pourcentage pondéré par l'occurence
    found_words = []
    for w1 in wordscountNew.keys():
        if w1 in wordscountB:
            found_words.append((w1,wordscountNew[w1]))
            
    print(np.round(sum([e[1] for e in found_words])*100, 2))
    
    return found_words

def maskSource(l):
    if 'UD' in l:
        return l.split('UD_')[1].split('-')[0]
    return l


#distances=best_distances.copy()
#Voir les distance entre représentation (distances) & one-to-one
def getMappingAvailability(charsA, charsB, distances, n_closest_chars=7):
    mapping_availability = 0
    cmap = dict()
    for y,k in enumerate(charsA):
        x = np.argsort(distances[y,:])[0:n_closest_chars]
        cmap[k] = [charsB[i] for i in x] 
    for c,closest_chars in cmap.items():
        mapping_availability += c in closest_chars
        #print(c,':', closest_chars, ':', c in closest_chars)
    mapping_availability = np.round(mapping_availability/len(cmap)*100,2)
    #print('mapping_availability:',mapping_availability)
    return mapping_availability

#Voir les distance entre représentation (distances) & one-to-one
def getMappingAvgRank(charsA, charsB, distances, n_most_frequent_analysis=20):
    avg_rank = 0
    cmap = dict()
    n_found = 0
    for y,k in enumerate(charsA):
        x = np.argsort(distances[y,:])
        cmap[k] = [charsB[i] for i in x].index(k) if k in charsB else -1
        #print(k,cmap[k], [charsB[i] for i in x][0:5])
        avg_rank += cmap[k] if cmap[k] != -1 else 0
        n_found += 1 if cmap[k] != -1 else 0
        if y >= n_most_frequent_analysis:
            break
    avg_rank = np.round(avg_rank/n_found,2)
    #print('avg_rank:',avg_rank)
    return avg_rank


#map: use the rank mapping on the matrices of langA
def create_mapped_mat(m, rank_mapping):
    
    m = m.copy() #copy the raw matrix
    sizepad_locations = m == -2
    mapped_mask = np.zeros(m.shape,dtype=bool)
    
    #for each character mapping
    for rA,rB in rank_mapping.items():
        #validate the mapping and map it
        mapped_mask[m == rA] = True
        m[m == rA] = -rB
    #put the mapping in positive
    m = -m
    #disable unmapped characters
    m[~mapped_mask] = -1
    m[sizepad_locations] = -2
    
    #count discarded character
    #n_discarded_char = sum([sum(m==-1)[0] if len(m) > 0 else 0 for m in mat_dict_mapped.values()])
    #n_total_char = sum([np.prod(m.shape) for m in mat_dict_mapped.values()])
    #p_discarded = np.round(n_discarded_char/n_total_char*100,2)
    
        
    return m#, p_discarded

#TODO unique better than that
#mat_langB, mapped_mat, computation_masks, current_cov_masks = mats[langB], mapped_mat, (validchar_maskA,validchar_maskB), (current_partialcov_mask,current_fullcov_mask)
def compute_coverages(mat_langB, mapped_mat, rank_to_charsB, len_intval = [4,99], \
                      computation_masks = (None,None), current_cov_masks = (None,None)):
    #compute score: compute potential word coverage in langB of mapped langA words
    #Init
    discarded = -1
    found_words = {}
    precomputed = {} 
    precomputedA = {} 
    lens = mapped_mat.shape[1] - (mapped_mat==-2).sum(-1)
    len_mask = (lens >= min(len_intval)) & (lens <= max(len_intval))
    is_covmask_defined = type(current_cov_masks[0]) != type(None)
    is_computation_mask_defined = type(computation_masks[0]) != type(None)
    
    #initialise la matrice des mots pré-exclus
    if is_covmask_defined: 
         partialcov_mask, fullcov_mask = current_cov_masks
    else: 
        partialcov_mask = np.ones(len(mapped_mat), dtype=bool); 
        fullcov_mask = np.zeros(len(mapped_mat), dtype=bool); 
    
    #applique les masques préliminaires de limitations de calculs
    matB = mat_langB
    fullcov_mask = fullcov_mask
    ids = range(0,len(mapped_mat))
    if is_computation_mask_defined:
        computation_maskA, computation_maskB = computation_masks
        maskA = computation_maskA & partialcov_mask
        mapped_mat = mapped_mat[maskA] #masque des mots ayant le caractère utile A 
        matB = mat_langB[computation_maskB] #masque des mots ayant le caractère utile B
        ids = maskA.nonzero()[0]
        
        
    
    #Pour chaque mot vecteur lA de la langue A
    i_compute = 0
    for ilA, lA in enumerate(mapped_mat):
        
        #tout est matché par défaut
        potential_matching = np.zeros(len(matB), dtype=bool)
        potential_matching[:] = True
        
        #word = ''.join([rank_to_charsB[r] for r in lA])
        
        matched = False
        if tuple(lA) in precomputedA:
            matched = precomputedA[tuple(lA)]
        else:
            i_compute += 1
            #Pour chaque caractère rang rA du mot lA
            for i_lA, rA in enumerate(lA):
                #si le caractère n'est pas à ignorer
                if rA != discarded:
                    #On vérifie le matching potentiel de tous les mots de B
                    if not (i_lA,rA) in precomputed:
                        precomputed[(i_lA,rA)] = matB[:,i_lA] == rA
                    mask = precomputed[(i_lA,rA)]
                    potential_matching = potential_matching & mask
                    
                    #on compte, si aucun matching, on cut
                    potential_matching_n = np.sum(potential_matching)
                    #print(potential_matching_n)
                    if potential_matching_n == 0:
                        break
            potential_matching_n = np.sum(potential_matching)
            matched = potential_matching.any()
            precomputedA[tuple(lA)] = matched
        
        #Si il y a un matching partiel
        if matched:
            #word = ''.join([rank_to_charsB[r] for r in lA])
            #word_cov = 1-(sum(lA == discarded) / len(lA))
            #found_words[word] = word_cov
            #Si le partiel est en réalité complet
            if sum(lA == discarded) == 0:
                fullcov_mask[ids[ilA]] = True
        else:
            partialcov_mask[ids[ilA]] = False
    #if len(mapped_mat) > 0:
    #   print(i_compute/len(mapped_mat),'% computations')
    
    #Calcule les moyenne de couvertures partielles et complètes du mapping
    partial_word_coverage = sum(partialcov_mask[len_mask]) / sum(len_mask)
    partial_word_coverage = np.round(partial_word_coverage*100, 6)
    full_word_coverage = sum(fullcov_mask[len_mask]) / sum(len_mask)
    full_word_coverage = np.round(full_word_coverage*100,6)
    
    return partial_word_coverage, full_word_coverage, found_words, partialcov_mask, fullcov_mask


#mat_langB, mapped_mat, rank_to_charsB, length_interval = mats[langB], mapped_mat, rank_to_charsB, len_intval
def basic_compute_coverages(mat_langB, mapped_mat, rank_to_charsB, length_interval = [4,99]):
    #compute score: compute potential word coverage in langB of mapped langA words
    #Init
    discarded = -1
    partial_word_coverage = 0
    full_word_coverage = 0
    n_word = 0
    found_words = {}
    precomputed = {} 
    
    #Pour chaque mot vecteur lA de la langue A
    for lA in mapped_mat:
        length = sum(lA != -2)
        #On ne fait les calculs que sur les intervalles de taille intéressants
        if length < min(length_interval) or length > max(length_interval):
            continue
        
        n_word += 1 
        #tout est matché par défaut
        potential_matching = np.zeros(len(mat_langB), dtype=bool)
        potential_matching[:] = True
        
        word = ''.join([rank_to_charsB[r] if r != -1 else '_' for r in lA])
        #print(word)
        #if word == 'Lincoln':
        #    assert True == False
        
        #Pour chaque caractère rang rA du mot lA
        for i_lA, rA in enumerate(lA):
            #si le caractère n'est pas à ignorer
            if rA != discarded:
                #On vérifie le matching potentiel de tous les mots de B
                if not (i_lA,rA) in precomputed:
                    precomputed[(i_lA,rA)] = mat_langB[:,i_lA] == rA
                mask = precomputed[(i_lA,rA)]
                potential_matching = potential_matching & mask
                
                ##print potential matching
                #idx = np.nonzero(potential_matching)
                #for lB in mB[idx]:
                #    w = ''.join([rank_to_charsB[r] if r != -1 else '_' for r in lB])
                #    print(w)
                #print(mB[idx])
                
                #on compte, si aucun matching, on cut
                potential_matching_n = np.sum(potential_matching)
                #print(potential_matching_n)
                if potential_matching_n == 0:
                    break
        potential_matching_n = np.sum(potential_matching)
        
        #if word == 'Lincoln':
        #   assert True == False
        
        #if word in wordsB:
        #    print(word,potential_matching_n)
        #Si il y a un matching partiel
        if potential_matching.any():
            partial_word_coverage += 1
            word = ''.join([rank_to_charsB[r] if r != discarded else '_' for r in lA])
            word_cov = 1-(sum(lA == discarded) / len(lA))
            found_words[word] = word_cov
            #Si le partiel est en réalité complet
            if sum(lA == discarded) == 0:
                full_word_coverage += 1
                #print(word)
    
    #Calcule les moyenne de couvertures partielles et complètes du mapping
    partial_word_coverage = partial_word_coverage / n_word
    partial_word_coverage = np.round(partial_word_coverage*100, 6)
    full_word_coverage = full_word_coverage / n_word
    full_word_coverage = np.round(full_word_coverage*100,6)
    
    return partial_word_coverage, full_word_coverage, found_words

#Build matrices of words
def build_words_mats(langs):
    #init
    chars_ranks, mats = {},{}
    max_h = 20000
    max_w = 20
    sizepad_id = -2
    
    for lang in langs:
        #dictionnaire taille => matrice
        words = loadWords(lang)
        wc,_ = countWords(words)
        mat = np.zeros((min(len(wc),max_h),max_w),dtype=np.int16)+sizepad_id
        
        #build chars rank
        chars,n_chars = countChars(list(words))
        chars_rank = {k:list(chars.keys()).index(k) for k in list(chars.keys())}
        chars_ranks[lang] = chars_rank
        
        #Pour chaque mot 
        for iw,w in enumerate(wc.keys()):
            #skip too long words
            if len(w) > max_w :
                continue
            #end
            if iw == max_h:
                break
            #remplissage de mat
            for jc,c in enumerate(w):
                mat[iw,jc] = chars_rank[c]
            
        #enregistrement matrices par langue
        mats[lang] = mat
    return mats, chars_ranks

#yield all combinations of lists parameters
def combinations(*lists):
    lists = lists[0]
    iterations = [0]*len(lists)
    #print(iterations, len(lists), lists)
    while True:
        if type(lists) == list:
            yield [lists[i][idx] for i,idx in enumerate(iterations)]
        elif type(lists) == dict:
            yield {k:lists[k][idx] for idx, k in zip(iterations, lists.keys())}
        iterations[0] += 1
        #print(iterations)
        for i in range(len(lists)):
            #print(i, iterations[i], lists[i])
            lenbase = None
            if type(lists) == list:
                lenbase = len(lists[i])
            elif type(lists) == dict:
                lenbase = len(lists[list(lists.keys())[i]])
            if iterations[i] == lenbase:
                if i == len(lists)-1:
                    return None
                iterations[i] = 0
                iterations[i+1] += 1

def getlang(corpus):
    return corpus.split('_')[1].split('-')[0] if corpus != 'voynich' else corpus

def getlangs(idres):
    if not 'voynich' in idres:
        return (idres.split('UD_')[1].split('-')[0], idres.split('UD_')[2].split('-')[0])
    elif idres.startswith('voynich'):
        return ('voynich',idres.split('UD_')[1].split('-')[0])
    elif 'voynich' in idres:
        return (idres.split('UD_')[1].split('-')[0], 'voynich')
        
        
        


