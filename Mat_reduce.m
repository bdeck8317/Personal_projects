%%

corrDir = 'C:\Users\Ben\Google Drive\CogNeW Laboratory Files\Pipelines\SVR_Analysis\MatchRate0.9/ROI2ROIFC_Indi/'; % Define your working folder
if ~isdir(corrDir)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(corrDir, '*_corr_z.mat');
matFiles = dir(filePattern);

%%
% load comms file
comms_path= 'C:\Users\Ben\Google Drive\CogNeW Laboratory Files\Pipelines\SVR_Analysis\MatchRate0.9/'
addpath(comms_path)
coms_nodes = open('comms_nodes.mat')
comms = coms_nodes.comms;


filePattern = fullfile(corrDir, '*_corr_z.mat');
matFiles = dir(filePattern);
for k = 1:length(matFiles)
  baseFileName = matFiles(k).name;
  fullFileName = fullfile(corrDir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  matData(k) = load(fullFileName);
end


%%
redNet = zeros(max(comms),max(comms));


for cordat = 1:length(matData)
    corrMat = matData(cordat).CorrMat
    for ii = 1:max(comms);
        for jj = 1:max(comms);
            net1 = find(comms == ii);
            net2 = find(comms ==jj);
            redNet(ii,jj) = mean(nonzeros(corrMat(net1, net2)));
            redNet2 = tril(redNet);
        end
    end
    reduce_struct(cordat).CorrMat= redNet2
end
        


%%
%for dd = 1:length(reduce_struct)
%    for di= 1:12
%        danwith = reduce_struct(di).CorrMat(6,6)
%        danbtw1 = reduce_struct(di).CorrMat(8,6)
%        danbtw2 = reduce_struct(di).CorrMat(13,6)
%        FPCNwith = reduce_struct(di).CorrMat(13,13)
%        FPCNbtw1 = reduce_struct(di).CorrMat(13,8)
%        FPCNbtw2 = reduce_struct(di).CorrMat(13,6)
%        SALwith = reduce_struct(di).CorrMat(8,8)
%        SALbtw1 = reduce_struct(di).CorrMat(8,6)
%        SALbtw2 = reduce_struct(di).CorrMat(13,8)
%        Subjectwise_connectivity(di,:) = [danwith danbtw1 danbtw2 FPCNwith FPCNbtw1 FPCNbtw2 SALwith SALbtw1 SALbtw2]
        
%    end   
%end

for dd = 1:length(reduce_struct)
    for di= 1:12
        handwith = reduce_struct(di).CorrMat(18,18)
        handbtw1 = reduce_struct(di).CorrMat(18,1)
        handbtw2 = reduce_struct(di).CorrMat(18,2)
        latviswith = reduce_struct(di).CorrMat(1,1)
        latvisbtw1 = reduce_struct(di).CorrMat(18,1)
        latvisbtw2 = reduce_struct(di).CorrMat(2,1)
        primviswith = reduce_struct(di).CorrMat(2,2)
        primvisbtw1 = reduce_struct(di).CorrMat(2,1)
        primvisbtw2 = reduce_struct(di).CorrMat(18,2)
        Subjectwise_connectivity(di,:) = [handwith handbtw1 handbtw2 latviswith latvisbtw1 latvisbtw2 primviswith primvisbtw1 primvisbtw2]
        
    end   
end

%%

  
Subjectwise_connectivity_trans = transpose(Subjectwise_connectivity)
Network_names = (["Within_HAND";"Between_Hand_LatVis";"Between_Hand_PrimVis";"Within_LatVis";"Between_LatVis_Hand";"Between_LatVis_Primary";"Within_PrimVis";"Between_PrimVis_LatVis";"Between_PrimVis_Hand"])
 
 T = table(Network_names,Subjectwise_connectivity_trans)
 writetable(T,'oddnetreducedconn.txt')
 
 Ttablearray=table2array(T);
 Ttabletransposed=array2table(Ttablearray.');
 writetable(Ttabletransposed,'ODDreducedmatconntrans.txt')
 
 %%
 writetable(Ttabletransposed, 'ODDnetant_connectivity_09.csv', 'Delimiter', ',')
 

 
