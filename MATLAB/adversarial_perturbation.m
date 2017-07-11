function adv = adversarial_perturbation(x,l,Df_base,f_out,opts)
NUM_LABELS = 10;
OS = 0.02;
Q = 2;
MAX_ITER = 100;
if(nargin==5)
    if isfield(opts,'labels_limit') NUM_LABELS = opts.labels_limit;end;
    if isfield(opts,'overshoot') OS = opts.overshoot;end;
    if isfield(opts,'norm_p') 
        Q = opts.norm_p/(opts.norm_p-1);
        if opts.norm_p==Inf
            Q = 1;
        end
    end
    if isfield(opts,'max_iter') MAX_ITER = opts.max_iter;end;
end

Df = @(y,idx) Df_base(y,l,idx); 

ff = f_out(x,0);
ff = ff-ff(l);
[~,I] = sort(ff,'descend');
labels = I(2:NUM_LABELS);

r = x*0;
x_u = x;

itr = 0;
while(f_out(x+(1+OS)*r,1)==l && itr<MAX_ITER)
    itr = itr + 1;
        
    ff = f_out(x_u,0);
    ff = ff-ff(l);
    
    idx = [l labels];
    ddf = Df(x_u,idx);
    
    dr = project_boundary_polyhedron(ddf,ff(idx),Q);
       
    x_u = x_u+dr;
    r = r + dr;
end

adv.r = (1+OS)*r;
adv.new_label = f_out(x+(1+OS)*r,1);
adv.itr = itr;
end

function dir = project_boundary_polyhedron(Df,f,Q)
res = abs(f)./arrayfun(@(idx) norm(Df(:,idx),Q), 1:size(Df,2));
[~,ii]=min(res);
if isinf(Q)
    dir = res(ii).*(abs(Df(:,ii))>=max(Df(:,ii))).*sign(Df(:,ii));
elseif(Q==1)
    dir = res(ii).*sign(Df(:,ii));
else
    dir = res(ii)*(abs(Df(:,ii))/norm(Df(:,ii),Q)).^(Q-1).*sign(Df(:,ii));
end
end
