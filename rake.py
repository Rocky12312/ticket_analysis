import re
import operator
import math
import os

debug = False
test = True

def isnum (s):
    try:
        if '.' in s:
            float(s)
        else:
            int(s)
        return True
    
    except ValueError:
        return False

#Here our main purpose is to create a list of stopwords
def loadStopWords(stopWordFile):
    stopWords = []
    for line in open(stopWordFile):
        if (line.strip()[0:1] != "#"):
            for word in line.split( ): #in case more than one per line
                stopWords.append(word)
    return stopWords


#Returning the words from the text(checking if there length is suitable)
def separatewords(text,minWordReturnSize):
    splitter=re.compile("[^a-zA-Z0-9_\\+\\-/]")
    words = []
    for singleWord in splitter.split(text):
        currWord = singleWord.strip().lower()
        if len(currWord)>minWordReturnSize and currWord != "" and not isnum(currWord): 
            words.append(currWord)
    return words

#Returning a list of sentences
def splitSentences(text):
    sentenceDelimiters = re.compile(u'[.!?,;:\t\\-\\"\\(\\)\\\'\u2019\u2013]')
    sentenceList = sentenceDelimiters.split(text)
    return sentenceList

def buildStopwordRegExPattern(pathtostopwordsfile):
    stopwordlist = loadStopWords(pathtostopwordsfile)
    stopwordregexlist = []
    for wrd in stopwordlist:
        wrdregex = '\\b' + wrd + '\\b'
        stopwordregexlist.append(wrdregex)
    stopwordpattern = re.compile('|'.join(stopwordregexlist), re.IGNORECASE)
    return stopwordpattern

def generateCandidateKeywords(sentenceList, stopwordpattern):
    phraseList = []
    for s in sentenceList:
        tmp = re.sub(stopwordpattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if (phrase!=""):
                phraseList.append(phrase)
    return phraseList

def calculateWordScores(phraseList):
    wordfreq = {}
    worddegree = {}
    for phrase in phraseList:
        wordlist = separatewords(phrase,0) 
        wordlistlength = len(wordlist)
        wordlistdegree = wordlistlength - 1
        #if wordlistdegree > 3,wordlistdegree = 3
        for word in wordlist:
            wordfreq.setdefault(word,0)
            wordfreq[word] += 1
            worddegree.setdefault(word,0)
            worddegree[word] += wordlistdegree #orig.
            #worddegree[word] += 1/(wordlistlength*1.0)
    for item in wordfreq:
        worddegree[item] = worddegree[item]+wordfreq[item] 	

# Calculate Word scores = deg(w)/frew(w)
    wordscore = {}
    for item in wordfreq:
        wordscore.setdefault(item,0)
        wordscore[item] = worddegree[item]/(wordfreq[item] * 1.0)
        #wordscore[item] = wordfreq[item]/(worddegree[item] * 1.0) 
    return wordscore

def generateCandidateKeywordScores(phraseList, wordscore):
    keywordcandidates = {}
    for phrase in phraseList:
        keywordcandidates.setdefault(phrase,0)
        wordlist = separatewords(phrase,0) 
        candidatescore = 0
        for word in wordlist:
            candidatescore += wordscore[word]
        keywordcandidates[phrase] = candidatescore
    return keywordcandidates

def rake(text):
    sentenceList = splitSentences(text)
    stoppath = os.path.join(os.path.dirname(__file__),"stop_words.txt")

    stopwordpattern = buildStopwordRegExPattern(stoppath)

    #Generatating candidate keywords
    phraseList = generateCandidateKeywords(sentenceList, stopwordpattern)

    #Calculating individual word scores
    wordscores = calculateWordScores(phraseList)

    #Generating candidate keyword scores
    keywordcandidates = generateCandidateKeywordScores(phraseList, wordscores)
    return keywordcandidates

