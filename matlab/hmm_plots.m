function hmm_plots(csvfile)
    data = readtable(csvfile);

    dates = data{:,2};
    values = data{:,3};

    plot(dates,values)
    grid on
    ylim([-4,4])
end