%%
%   MATLAB code for DeepFool
%
%   adversarial_DeepFool_caffe(x,net):
%   computes the adversarial perturbations for a Caffe's model
%
%   INPUTS 
%   x: image in W*H*C format
%   net: Caffe's network (without loss layer - do not forget to enable 'force_backward')
%   opts: A struct contains parameters (see README)
%   OUTPUTS
%   r_hat: minimum perturbation
%   l_hat: adversarial label
%   l: classified label
%   itr: number of iterations
%
%   please cite: S. Moosavi-Dezfooli, A. Fawzi, P. Frossard: DeepFool: a simple and accurate method to fool deep neural networks.
%                In Computer Vision and Pattern Recognition (CVPR 2016), IEEE, 2016.
%%
function [r_hat,l_hat,l,itr] = adversarial_DeepFool_caffe(x,net,opts)
size_x = size(x);
x = reshape(x,numel(x),1);
l = f(x,1);

if(nargin==3)
    adv = adversarial_perturbation(x,l,@Df,@f,opts);
else
    adv = adversarial_perturbation(x,l,@Df,@f);
end

l_hat = adv.new_label;
r_hat = reshape(adv.r,size_x);
itr = adv.itr;

    function out = f(y,flag)
        y = reshape(y,size_x);
        
        out = net.forward({y}); %do forward pass
        out = out{1}'; %convert 'out' from a cell array to a matrix
        
        %flag==0:compute the outputs
        %flag==1:compute the label
        if flag==1
            [~,out] = max(out);
        end
    end


    function dzdx = Df(y,label,idx)
        y = reshape(y,size_x);
        net.forward({y}); %do forward pass
        
        for i=1:numel(idx)
            dzdy = zeros(net.blobs(net.blob_names{end}).shape,'single');
            
            dzdy(idx(i)) = 1;
            
            res = net.backward({dzdy}); %do backward pass
            dzdx(:,i) = reshape(res{1},numel(y),1);
        end
        dzdx = dzdx-repmat(dzdx(:,idx==label),1,numel(idx));
    end
end
