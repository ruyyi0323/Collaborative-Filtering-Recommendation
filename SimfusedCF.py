import pandas as pd
import numpy as np

def readfiles():
    ratings = pd.DataFrame.from_csv('/Users/ruyyi/PycharmProjects/CollaborativeFiltering/ratings.csv', index_col=None)
    index = list(range(ratings.shape[0]))
    recordsize = max(index)
    ratings['index'] = index
    ratings = ratings.reindex_axis(['index'] + list(ratings.columns[:4]), axis=1)
    print('reading dataset complete, record size is',recordsize)
    return recordsize, ratings

def basicinfo(ratings):
    movieid = ratings['movieId']
    moviesize = (np.unique(movieid.sort_values())).shape[0]
    userid = ratings['userId']
    usersize = (np.unique(userid.sort_values())).shape[0]
    print('reading dataset complete, user size is',usersize,'movie size is', moviesize)
    return usersize,moviesize

def idmodify(usersize,moviesize,ratings):
    movieid = ratings['movieId']
    movieid_index = list(range(moviesize))
    movieid_list = np.unique(movieid.sort_values())
    movieid_indexdict = dict(list(zip(list(movieid_list),movieid_index)))
    movieid_new = movieid.map(movieid_indexdict)
    ratings['movieId'] = movieid_new
    userid = ratings['userId']
    userid_index = list(range(usersize))
    userid_list = np.unique(userid.sort_values())
    userid_indexdict = dict(list(zip(list(userid_list),userid_index)))
    userid_new = userid.map(userid_indexdict)
    ratings['userId'] = userid_new
    return ratings

def splitdataset(portion,recordsize,ratings):
    #portion must be 0-1
    ratings = ratings.sample(frac = 1)
    testset = ratings[1:int(portion*recordsize)]
    trainset = ratings[int(portion*recordsize):]
    print('split dataset complete with trainset size is',trainset.shape[0],'testset size is',testset.shape[0])
    return trainset,testset

def createratingmatrix(ratings,usersize,moviesize):
    rating_matrix = np.zeros((usersize,moviesize))
    for index,row in ratings.iterrows():
        #userid = int(row['userId'])
        #movieid = int(row['movieId'])
        #rating = row['rating']
        rating_matrix[int(row['userId'])][int(row['movieId'])] = row['rating']
    print('building rating matrix complete')
    return rating_matrix

def userbased_createsimilaritymatrix(ratings):
    from sklearn.metrics.pairwise import cosine_similarity
    user_similarity = cosine_similarity(ratings, ratings)
    print('user similarity matrix build complete with shape',user_similarity.shape)
    return user_similarity

def itembased_createsimilaritymatrix(ratings):
    from sklearn.metrics.pairwise import cosine_similarity
    item_similarity = cosine_similarity(np.transpose(ratings), np.transpose(ratings))
    print('item similarity matrix build complete with shape',item_similarity.shape)
    return item_similarity

def findneighbors(userid,movieid,user_similaritymatrix,item_similaritymatrix,ratingmatrix,k):
    usersimilarityvector = user_similaritymatrix[:, userid]
    moviesimilarityvector = item_similaritymatrix[:, movieid]
    userratingvector = ratingmatrix[:,movieid]
    movieratingvector = ratingmatrix[userid,:]
    useridlist = list(range(user_similaritymatrix.shape[0]))
    movieidlist = list(range(item_similaritymatrix.shape[0]))
    userneighbor = pd.DataFrame(data = {'userid' : useridlist,
                                            'rating' : userratingvector,
                                            'similarity' : usersimilarityvector},
                                    index = useridlist)
    movieneighbor = pd.DataFrame(data = {'movieid' : movieidlist,
                                            'rating' : movieratingvector,
                                            'similarity' : moviesimilarityvector},
                                    index = movieidlist)
    userneighbor = userneighbor.loc[userneighbor['rating'] != 0].sort_values( by = 'similarity',ascending = False)
    if userneighbor.shape[0] > k:
        userneighbor = userneighbor.iloc[:k]
    movieneighbor = movieneighbor.loc[movieneighbor['rating'] != 0].sort_values(by = 'similarity',ascending = False)
    if movieneighbor.shape[0] > k:
        movieneighbor = movieneighbor.iloc[:k]
    simfuseneighbor = getfuseneighbor(userid, movieid,
                                      userneighbor, movieneighbor,
                                      ratingmatrix,
                                      user_similaritymatrix,item_similaritymatrix)
    simfuseneighbor = simfuseneighbor.loc[simfuseneighbor['rating'] != 0]
    return userneighbor,movieneighbor,simfuseneighbor

def getfuseneighbor(userid,movieid,userneighbor,movieneighbor,ratingmx,usimmx,isimmx):
    simfuseneighbor = pd.DataFrame(columns=['userid','movieid','rating','similarity'])
    for _,urow in userneighbor.iterrows():
        for _,mrow in movieneighbor.iterrows():
            uid = int(urow['userid'])
            mid = int(mrow['movieid'])
            rate = ratingmx[uid][mid]
            usim = usimmx[userid][uid]
            isim = isimmx[movieid][mid]
            row = pd.DataFrame([dict(userid = uid,
                                     movieid = mid,
                                     rating = rate,
                                     similarity = 1/np.sqrt(pow((1/usim),2)+pow((1/isim),2)))])
            simfuseneighbor = simfuseneighbor.append(row,ignore_index=True)
    return simfuseneighbor

def initiaterecommendation(testset):
    k,alpha,beta = 10,0.7,0.7
    result = testset[['userId','movieId','rating']]
    #result['userbased_pred'],result['itembased_pred'],result['fuse_pred'] = 0,0,0
    userbased_pred,itembased_pred,fuse_pred = [],[],[]
    return k,alpha,beta,result,userbased_pred,itembased_pred,fuse_pred

def userbasedrecommendation(userneighbor):
    pred = np.sum(userneighbor['rating']*userneighbor['similarity']) / np.sum(userneighbor['similarity'])
    return pred

def itembasedrecommendation(movieneighbor):
    if np.sum(movieneighbor['similarity']) != 0:
        pred = np.sum(movieneighbor['rating']*movieneighbor['similarity']) / np.sum(movieneighbor['similarity'])
    else:pred = 3
    return pred

def fusedrecommendation(simfuseneighbor):
    pred = np.sum(simfuseneighbor['rating']*simfuseneighbor['similarity']) / np.sum(simfuseneighbor['similarity'])
    return pred

if __name__ == "__main__" :
    recordsize, ratings = readfiles()
    usersize, moviesize = basicinfo(ratings)
    ratings = idmodify(usersize,moviesize,ratings)
    trainset,testset = splitdataset(0.01,recordsize,ratings)
    ratingmatrix = createratingmatrix(trainset,usersize,moviesize)
    userbased_similaritymatrix = userbased_createsimilaritymatrix(ratingmatrix)
    itembased_similaritymatrix = itembased_createsimilaritymatrix(ratingmatrix)

    #recommendation start
    k, alpha, beta, result, userbased_pred, itembased_pred, fuse_pred = initiaterecommendation(testset)
    for _, row in result.iterrows():
        userid = int(row['userId'])
        movieid = int(row['movieId'])
        userneighbor,movieneighbor,simfuseneighbor = findneighbors\
            (userid,movieid,userbased_similaritymatrix,itembased_similaritymatrix,ratingmatrix,k)

        upred,ipred,fpred = 3,3,3
        if userneighbor.shape[0] != 0:
            upred = userbasedrecommendation(userneighbor)
            userbased_pred.append(upred)
        else:alpha = 0
        if movieneighbor.shape[0] != 0:
            ipred = itembasedrecommendation(movieneighbor)
        else:beta = 0
        if simfuseneighbor.shape[0] != 0:
            fpred = fusedrecommendation(simfuseneighbor)*(1-alpha)*(1-beta)+\
                    upred*alpha+ipred*beta

        userbased_pred.append(upred)
        itembased_pred.append(ipred)
        fuse_pred.append(fpred)

        print('recommandation complete for user',userid,'on movie',movieid,upred,ipred,fpred)




    print('done')