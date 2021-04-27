function hmm_plots(ticker_str)
    sign(11)
    sign(-11)
    csvfile = append(ticker_str,'.csv');
    data = readtable(csvfile);

    dates = data{:,2};
    open_values = data{:,3};
    frac_pred = data{:,4};
    predictions = data{:,5};
    close_values = data{:,6};
    
    value_change = (close_values - open_values);

    for c = 1:size(dates)
        actual_frac(c) = value_change(c)/open_values(c);
    end

    figure
    plot(dates,predictions, 'bo-')
    grid on
    hold on
    plot(dates,close_values, 'r*-')
    title(append(ticker_str,' Stock Price Predictions 2021'), 'FontSize', 24)
    legend('Predicted','Actual', 'FontSize', 24)
    ax = gca;
    ax.FontSize = 20; 
    hold off
    
    figure
    plot(dates,frac_pred,'r-o')
    grid on
    hold on
    plot(dates,actual_frac,'b-*')
    title(append(ticker_str,' Stock Price Fractional Change Predictions 2021'), 'FontSize', 24)
%     x = [dates(1), dates(end)];
%     y = [0,0];
%     plot(x,y,'k', 'LineWidth', 2)
    legend('Predicted','Actual', 'FontSize', 24)
    ax = gca;
    ax.FontSize = 20; 
    hold off
    
    correct_counter = 0;
    incorrect_counter = 0;
    
    for c=1:size(dates)
       if sign(frac_pred(c)) == sign(actual_frac(c))
          correct_counter = correct_counter+1;
       else
          incorrect_counter = incorrect_counter+1;
       end
    end
    
%     checking for correctness of discrete obs with two cases
%     for c=1:size(dates)
%        if (frac_pred(c)==1 && sign(actual_frac(c)) > 0) || (frac_pred(c)==0 && sign(actual_frac(c)) < 0)
%           correct_counter = correct_counter+1;
%        else
%           incorrect_counter = incorrect_counter+1;
%        end
%     end
%     
    %     checking for correctness of discrete obs with more than two cases
%     for c=1:size(dates)
%        if (frac_pred(c)>0 && sign(actual_frac(c)) > 0) || (frac_pred(c)<=0 && sign(actual_frac(c)) < 0)
%           correct_counter = correct_counter+1;
%        else
%           incorrect_counter = incorrect_counter+1;
%        end
%     end
    
    x = size(dates);
    percentage = 100*correct_counter / x(1:1);
    append("This model is correct in its change of directoin ",string(correct_counter)," times: It is wrong ",string(incorrect_counter)," times")
    append(string(percentage),"% accurate")
    
    for c=1:size(dates)
       if frac_pred(c) > 0
          money_made(c) = 1*(close_values(c) - open_values(c));
       elseif frac_pred(c) <= 0 
          money_made(c) = 1*(open_values(c) - close_values(c));
       end
    end
    total = cumsum(money_made);
    figure
    plot(dates,money_made,'k-o')
    hold on
    title(append(ticker_str,' Money Made'), 'FontSize', 24)
    plot(dates,total,'r-*')
    hold off
    legend('Money Made That Day','Cumulative', 'FontSize', 24)
    ax = gca;
    ax.FontSize = 20; 
    append('$',append(string(total(end)),' dollars have been made per share'))
end