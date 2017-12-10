Search.setIndex({docnames:["core","identity_data_processor","index","modules","predict_identity_char_model","server","train_generic_identity_char_model","train_personalized_identity_char_model"],envversion:53,filenames:["core.rst","identity_data_processor.rst","index.rst","modules.rst","predict_identity_char_model.rst","server.rst","train_generic_identity_char_model.rst","train_personalized_identity_char_model.rst"],objects:{"":{core:[0,0,0,"-"],identity_data_processor:[1,0,0,"-"],server:[5,0,0,"-"]},"core.common_helpers":{FileHelper:[0,1,1,""],StringHelper:[0,1,1,""],TimeHelper:[0,1,1,""],TorchHelper:[0,1,1,""]},"core.common_helpers.FileHelper":{read_file:[0,2,1,""],read_file_as_ascii:[0,2,1,""],read_lines:[0,2,1,""],read_lines_as_ascii:[0,2,1,""],read_lines_as_ascii_iterator:[0,2,1,""],read_lines_iterator:[0,2,1,""]},"core.common_helpers.StringHelper":{ascii_from_unicode:[0,2,1,""],ascii_from_unicode_transliteration:[0,2,1,""],is_ascii_character:[0,2,1,""],tokenize_words:[0,2,1,""],tokenize_words_iterator:[0,2,1,""]},"core.common_helpers.TimeHelper":{time_delta:[0,2,1,""],time_since:[0,2,1,""]},"core.common_helpers.TorchHelper":{create_adam_optimizer:[0,2,1,""],create_cross_entropy_loss:[0,2,1,""],create_sgd_optimizer:[0,2,1,""],load:[0,2,1,""],prevent_exploding_gradient:[0,2,1,""],save:[0,2,1,""],set_seed:[0,2,1,""]},"core.corpus":{Corpus:[0,1,1,""]},"core.corpus.Corpus":{create_batch:[0,3,1,""],create_batches:[0,3,1,""],create_random_batch:[0,3,1,""],load:[0,2,1,""],save:[0,3,1,""],split_batched_data:[0,3,1,""],test:[0,4,1,""],test_file_path:[0,4,1,""],train:[0,4,1,""],train_file_path:[0,4,1,""],validation:[0,4,1,""],validation_file_path:[0,4,1,""],vocabulary:[0,4,1,""]},"core.data_processor":{DataProcessor:[0,1,1,""],PennTreeBankProcessor:[0,1,1,""]},"core.data_processor.PennTreeBankProcessor":{character_items_from_string:[0,3,1,""],process_corpus_as_characters:[0,3,1,""],process_corpus_as_words:[0,3,1,""],word_items_from_string:[0,3,1,""]},"core.prediction":{CharacterRNNPredictor:[0,1,1,""],RNNPredictor:[0,1,1,""]},"core.prediction.CharacterRNNPredictor":{predict_chars:[0,3,1,""],predict_sentence:[0,3,1,""],predict_sentences:[0,3,1,""],predict_words:[0,3,1,""],tensor_from_item:[0,3,1,""]},"core.prediction.RNNPredictor":{create_state:[0,3,1,""],next_state:[0,3,1,""],next_state_given_item:[0,3,1,""],predict:[0,3,1,""],predict_iterator:[0,3,1,""],predict_many_next:[0,3,1,""],predict_many_next_indices:[0,3,1,""],predict_next:[0,3,1,""],predict_next_index:[0,3,1,""],predict_while:[0,3,1,""],predict_while_iterator:[0,3,1,""]},"core.rnn":{CharacterRNN:[0,1,1,""],PersonalizedCharacterRNN:[0,1,1,""],RNN:[0,1,1,""],RNNConfig:[0,1,1,""]},"core.rnn.CharacterRNN":{config:[0,4,1,""],decoder:[0,4,1,""],dropout:[0,4,1,""],encoder:[0,4,1,""],forward:[0,3,1,""],rnn:[0,4,1,""]},"core.rnn.PersonalizedCharacterRNN":{config:[0,4,1,""],decoder:[0,4,1,""],dropout:[0,4,1,""],encoder:[0,4,1,""],rnn:[0,4,1,""],transfer_from_rnn_with_extra_layers:[0,5,1,""]},"core.rnn.RNN":{config:[0,4,1,""],create_hidden:[0,3,1,""],decoder:[0,4,1,""],dropout:[0,4,1,""],encoder:[0,4,1,""],forward:[0,3,1,""],rnn:[0,4,1,""]},"core.rnn.RNNConfig":{copy:[0,3,1,""],dropout:[0,4,1,""],embedding_size:[0,4,1,""],from_serialized:[0,5,1,""],hidden_size:[0,4,1,""],init_weight_randomly:[0,4,1,""],layer_count:[0,4,1,""],serialize:[0,3,1,""],term_count:[0,4,1,""],tie_weights:[0,4,1,""],type:[0,4,1,""]},"core.vocabulary":{Vocabulary:[0,1,1,""]},"core.vocabulary.Vocabulary":{add:[0,3,1,""],copy:[0,3,1,""],count:[0,3,1,""],create_item_index_map:[0,2,1,""],eos_item:[0,3,1,""],from_list:[0,5,1,""],from_string:[0,5,1,""],index_from_item:[0,3,1,""],indices_from_items:[0,3,1,""],is_eos_item:[0,3,1,""],is_numeric_item:[0,3,1,""],is_unknown_item:[0,3,1,""],item_exists:[0,3,1,""],item_from_index:[0,3,1,""],items:[0,3,1,""],items_from_indices:[0,3,1,""],load:[0,2,1,""],long_tensor_from_items:[0,3,1,""],numeric_item:[0,3,1,""],one_hot_matrix_tensor_from_items:[0,3,1,""],one_hot_tensor_from_item:[0,3,1,""],remove_all:[0,3,1,""],save:[0,3,1,""],unknown_item:[0,3,1,""]},"identity_data_processor.IdentityDataProcessor":{items_from_string:[1,3,1,""],normalize_sentence:[1,3,1,""],normalize_string:[1,3,1,""],process_corpus:[1,3,1,""],process_overfit_corpus:[1,3,1,""]},"server.main":{about:[5,6,1,""],details:[5,6,1,""],file_exists:[5,6,1,""],index:[5,6,1,""],is_float:[5,6,1,""],load:[5,6,1,""],load_model:[5,6,1,""],load_vocabulary:[5,6,1,""],normalize_context:[5,6,1,""],normalize_identity:[5,6,1,""],normalize_temperatures:[5,6,1,""],parse_context:[5,6,1,""],parse_identity:[5,6,1,""],parse_sentence_prediction_request:[5,6,1,""],parse_temperatures:[5,6,1,""],predict_sentences:[5,6,1,""],predict_sentences_json:[5,6,1,""],prediction:[5,6,1,""],raise_if_file_not_exists:[5,6,1,""],sentences_prediction:[5,6,1,""],setup:[5,6,1,""],valid_identity:[5,6,1,""],valid_temperature:[5,6,1,""],valid_tempratures:[5,6,1,""]},core:{common_helpers:[0,0,0,"-"],corpus:[0,0,0,"-"],data_processor:[0,0,0,"-"],prediction:[0,0,0,"-"],rnn:[0,0,0,"-"],vocabulary:[0,0,0,"-"]},identity_data_processor:{CharDataProcessor:[1,1,1,""],IdentityDataProcessor:[1,1,1,""]},server:{main:[5,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","staticmethod","Python static method"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","classmethod","Python class method"],"6":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:staticmethod","3":"py:method","4":"py:attribute","5":"py:classmethod","6":"py:function"},terms:{"1x1":0,"case":0,"class":[0,1],"default":0,"float":[0,5],"function":[0,1],"int":0,"long":0,"return":[0,1],"static":0,"true":0,"while":0,The:[0,1,5],abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz:0,about:5,accumul:0,adam:0,add:0,adding:0,all:0,allow:0,amount:0,ani:0,api:5,arrai:5,ascii:0,ascii_from_unicod:0,ascii_from_unicode_transliter:0,ascii_lett:0,ascii_list:0,autograd:0,bank:0,base:[0,1],batch:0,batch_siz:0,befor:0,bool:0,calcul:0,call:0,charact:[0,1,5],character_items_from_str:0,characterrnn:0,characterrnnpredictor:0,chardataprocessor:1,cirterion:0,classmethod:0,clip_grad_max_norm:0,collect:0,comma:5,common:0,common_help:3,compos:0,comput:0,condit:0,config:0,configur:0,consid:0,consolid:5,construct:0,contain:0,content:3,context:[0,5],convers:0,convert:0,copi:0,core:3,corpu:[1,3],correspond:0,count:0,creat:0,create_adam_optim:0,create_batch:0,create_cross_entropy_loss:0,create_hidden:0,create_item_index_map:0,create_random_batch:0,create_sgd_optim:0,create_st:0,criterion:0,cross:0,crossentropyloss:0,cuda:0,current:0,data:[0,1,5],data_processor:3,dataprocessor:0,decod:0,def:0,defin:[0,1],delta:0,detail:5,details_file_path:5,determin:[0,5],dict:0,dictionari:0,doe:[0,5],dropout:0,each:0,embed:0,embedding_s:0,encod:0,end:0,end_of_sequence_token:[0,1],entri:5,entropi:0,eos:0,eos_item:0,error:5,eval_batch_s:[0,1],eval_mod:0,evalu:0,everi:0,exampl:0,exist:[0,5],extra:0,fail:0,fail_on_no_upd:0,fals:[0,1],file:[0,5],file_exist:5,file_path:[0,5],filehelp:0,first:0,flask:5,floattensor:0,follow:0,forward:0,found:0,from:[0,5],from_list:0,from_seri:0,from_str:0,gener:0,get:0,given:[0,5],gpu:0,gru:0,hidden:0,hidden_s:0,homer_model_file_path:5,hot:0,http:5,ident:[1,5],identity_data_processor:3,identitydataprocessor:1,includ:0,index:[0,2,5],index_from_item:0,indic:0,indices_from_item:0,indicesgiven:0,init_weight_randomli:0,initi:0,input:[0,1],instanc:0,is_ascii_charact:0,is_eos_item:0,is_float:5,is_numeric_item:0,is_unknown_item:0,item:[0,1],item_exist:0,item_from_index:0,item_index_map:0,items_from_indic:0,items_from_str:1,iter:0,its:[0,1],join:1,kei:0,languag:0,layer:0,layer_count:0,learn:0,learning_r:0,length:0,letter:0,line:0,linear:0,list:[0,1],load:[0,5],load_model:5,load_vocabulari:5,long_tensor_from_item:0,longtensor:0,loss:0,lower:0,lstm:0,main:3,make:0,mani:0,map:0,matrix:0,max_word_count:0,maximum:0,minut:0,mode:0,model:[0,5],model_file_path:5,modul:[2,3],move:0,multipl:0,nerual:0,network:0,neural:0,next:0,next_item:0,next_stat:0,next_state_given_item:0,none:[0,1],normal:[1,5],normalize_context:5,normalize_ident:5,normalize_sent:1,normalize_str:1,normalize_temperatur:5,not_end_of_word:0,note:0,number:0,numer:0,numeric_item:0,numeric_token:[0,1],obj:0,object:[0,1],offset:0,one:0,one_hot_matrix_tensor_from_item:0,one_hot_tensor_from_item:0,oper:0,optim:0,option:0,order:0,otherwis:0,output:0,overfit:1,packag:3,page:2,param:[],paramet:[0,1],pars:[0,1,5],parse_context:5,parse_ident:5,parse_sentence_prediction_request:5,parse_temperatur:5,path:[0,5],penn:0,penntreebankprocessor:0,per:0,perform:0,person:0,personalizedcharacterrnn:0,pickl:5,point:5,predict:[3,5],predict_char:0,predict_identity_char_model:3,predict_iter:0,predict_many_next:0,predict_many_next_indic:0,predict_next:0,predict_next_index:0,predict_sent:[0,5],predict_sentences_json:5,predict_whil:0,predict_while_iter:0,predict_word:0,predictor:0,preprocess:1,present:0,prevent_exploding_gradi:0,print:0,prior:0,process:[0,1],process_corpu:1,process_corpus_as_charact:0,process_corpus_as_word:0,process_corpus_join_as_train:[],process_overfit_corpu:1,processor:[0,1],provid:[0,5],rais:5,raise_if_file_not_exist:5,random:0,randomli:0,rate:0,read:0,read_fil:0,read_file_as_ascii:0,read_lin:0,read_lines_as_ascii:0,read_lines_as_ascii_iter:0,read_lines_iter:0,recurr:0,refer:0,remov:0,remove_al:0,repres:0,represent:0,request:5,resolv:0,rest:5,retriev:0,rnn:[3,5],rnnconfig:0,rnnpredictor:0,rout:5,save:0,search:2,second:0,seed:0,sentenc:[0,1,5],sentences_predict:5,separ:5,sequenc:0,sequence_length:[0,1],serial:0,server:3,set:[0,1,5],set_se:0,setup:5,sgd:0,sherlock_model_file_path:5,should:0,sinc:0,singl:1,size:0,sourc:[0,1,5],space:0,split:0,split_batched_data:0,start:0,state:0,statel:0,store:0,str:[0,1],string:[0,1,5],stringhelp:0,submodul:3,support:5,target:0,temperatur:[0,5],tensor:0,tensor_from_item:0,term:0,term_count:0,test:[0,1],test_file_path:0,thi:0,tie_weight:0,tied:0,time:0,time_delta:0,time_sinc:0,timehelp:0,tokenize_word:0,tokenize_words_iter:0,torch:0,torch_object:0,torchhelp:0,train:[0,1],train_batch_s:[0,1],train_file_path:0,train_generic_identity_char_model:3,train_personalized_identity_char_model:3,transfer:0,transfer_from_rnn_with_extra_lay:0,transliter:0,tree:0,tupl:0,two:0,type:[0,1],unicod:0,unit:0,unknown:0,unknown_item:0,unknown_token:0,updat:[0,1],upper:0,use:[0,5],use_cuda:[0,1,5],used:[0,1],using:0,valid:[0,1,5],valid_ident:5,valid_temperatur:5,valid_tempratur:5,validation_file_path:0,valu:[0,5],variabl:0,vocabulari:[1,3,5],vocabulary_file_path:5,vocuabulari:0,weight:0,when:0,where:0,whether:[0,5],which:0,within:5,word:0,word_items_from_str:0,yield:0,zero:0},titles:["core package","identity_data_processor module","Welcome to Identity word completion\u2019s documentation!","style-transfer","predict_identity_char_model module","server package","train_generic_identity_char_model module","train_personalized_identity_char_model module"],titleterms:{common_help:0,complet:2,content:[0,5],core:0,corpu:0,data_processor:0,document:2,ident:2,identity_data_processor:1,indic:2,main:5,modul:[0,1,4,5,6,7],packag:[0,5],predict:0,predict_identity_char_model:4,rnn:0,server:5,style:3,submodul:[0,5],tabl:2,train_generic_identity_char_model:6,train_personalized_identity_char_model:7,transfer:3,vocabulari:0,welcom:2,word:2}})