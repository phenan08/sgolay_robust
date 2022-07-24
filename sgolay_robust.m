function [y1,y2,y3] = sgolay_robust_2(y_input,x_input,half_window,order,step,opts)

arguments % Argument validation
    y_input         double  {mustBeNumeric,mustBeReal,mustBeVector} 
    x_input         double  {mustBeNumeric,mustBeReal,mustBeVector,mustBeEqualSize(y_input,x_input)}
    half_window  (1,1)   double  {mustBeNumeric,mustBeReal,mustBePositive}
    order   (1,1)   double  {mustBeNumeric,mustBeReal,mustBeGreaterThanOrEqual(order,0)}
    step    (1,1)   double  {mustBeNumeric,mustBeReal,mustBePositive}
    opts.Interpolation  string   {mustBeTextScalar,mustBeMember(opts.Interpolation,["linear","cubic","spline","nearest","makima"])} = "linear"
    opts.RobustWeightFunction  string   {mustBeTextScalar,mustBeMember(opts.RobustWeightFunction,["bisquare"])} = "bisquare"
    opts.DisplayWarnings  string   {mustBeTextScalar,mustBeMember(opts.DisplayWarnings,["on","off"])} = "on"
    opts.TolFun = 1e-3 ;
    opts.TolX = 1e-3 ;
    opts.MaxIter = 200 ;
end

% Disable warnings if requested
if opts.DisplayWarnings == "off"
    warning("off","all") ;
end
% When appropriate, w is rounded to nearest integer
if mod(half_window,1) ~= 0
    half_window = round(half_window) ;
    warning("Parameter w was rounded to nearest integer.") ;
end
% When appropriate, o is rounded to nearest integer
if mod(order,1) ~= 0
    order = round(order) ;
    warning("Parameter o was rounded to nearest integer.") ;
end
% When appropriate, step is rounded to nearest integer
if mod(step,1) ~= 0
    step = min(1,round(step)) ;
    warning("Step size was rounded to nearest integer.") ;
end

% Ckeck for NaNs
test = sum(isnan(y_input)) + sum(isnan(x_input)) ;
if test > 0
    error("The data supplied contain NaNs. Aborted.") ;
    return
end

% Check input vector types
test = isrow(x_input) ;
if test == true
    x_input_type = "row" ;
    x_input = x_input' ;
else
    x_input_type = "column" ;
end
test = isrow(y_input) ;
if test == true
    y_input_type = "row" ;
    y_input = y_input' ;
else
    y_input_type = "column" ;
end

% Memory preallocation
y1 = zeros(size(x_input)) ;
y2 = zeros(size(x_input)) ;
y3 = zeros(numel(x_input),order+1) ;


N = numel(y_input) ;

for i = 1:step:N

    ind1 = max(1,i-half_window) ;
    ind2 = min(N,i+half_window) ;
    x_spl = x_input(ind1:ind2) ;
    y_spl = y_input(ind1:ind2) ;

    weights = ones(size(y_spl)) ;

    p = inf(order+1,1) ; s = inf ; ds = 1 ; dp = 1 ;

    X = repmat(x_spl,1,order+1).^[order:-1:0] ;
    c = 4.685 ; nb_iter = 0 ;

    while (abs(ds) >= opts.TolFun) || (abs(dp) >= opts.TolX)

        if nb_iter > opts.MaxIter
            break
        end

        nb_iter = nb_iter+1 ;

        old_p = p ;
        p = (sqrt(weights).*X)\(sqrt(weights).*y_spl) ;
    
        y_spl_calc = X*p ;
        r = y_spl_calc-y_spl ;
        tau = median(abs(r-median(r)))/0.6745 ;
        if tau == 0
            s = std(r)*sqrt(1+1/numel(y_spl)+(x_input(i)-mean(x_spl)).^2/sum((x_spl-mean(x_spl)).^2)) ;
            break
        end
        z = r/tau ;
        weights = (abs(z)<c).*(1-(z/c).^2).^2 ;
        weights = weights.*(weights>=0) ; 
        old_s = s ;
        s = std(r,weights)*sqrt(1+1/numel(y_spl)+(x_input(i)-sum(weights.*x_spl)./sum(weights)).^2/sum((x_spl-sum(weights.*x_spl)./sum(weights)).^2)) ;
        dp = max((old_p-p)./p) ;
        ds = (old_s-s)/s ;

    end

    y3(i,:) = p' ;
    y2(i) = s ;
    y1(i) = polyval(p,x_input(i)) ;
 
end

if y_input_type == "row"
    y1 = y1' ;
    y2 = y2' ;
    x3 = y3' ;
    x_input = x_input' ;
    y_input = y_input' ;
end

if step == 1
    % No need for interpolation.
    return
end

y1 = interp1(x_input(1:step:N),y1(1:step:N),x_input,opts.Interpolation,"extrap") ;
y2 = interp1(x_input(1:step:N),y2(1:step:N),x_input,opts.Interpolation,"extrap") ;
y3 = interp1(x_input(1:step:N),y3(1:step:N,:),x_input,opts.Interpolation,"extrap") ;

end

function mustBeEqualSize(a,b)
    % Test for equal size
    if ~isequal(size(a),size(b))
        eid = 'Size:notEqual';
        msg = 'Size of first input must equal size of second input.';
        throwAsCaller(MException(eid,msg))
    end
end