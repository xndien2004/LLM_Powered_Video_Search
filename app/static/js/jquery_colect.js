$(document).ready(function() {
    $('.search-box-header').hover(
        function() {
            $('.actions').stop(true, true).show();
        },
        function() {
            setTimeout(function() {
                if (!$('.actions:hover').length) {
                    $('.actions').stop(true, true).hide();
                }
            }, 200);
        }
    );

    $('.actions').hover(
        function() {
            $(this).stop(true, true).show();
        },
        function() {
            $(this).stop(true, true).hide();
        }
    );

    $(document).click(function(event) {
        if (!$(event.target).closest('.search-box-header, .actions').length) {
            $('.actions').hide();
        }
    });
});


