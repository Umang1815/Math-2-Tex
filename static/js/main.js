$(document).ready(
    function () {

    var result_data = "";
    // Init
    $('.preview_section').hide();
    $('.loader').hide();
    $('#result').hide();

    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
      })
    

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#preview_image').css('content', 'url(' + e.target.result + ')');
                $('#preview_image').hide();
                $('#preview_image').fadeIn(300);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.upload_section').hide();
        $('.preview_section').show();
        $('.preview_section_right_2').hide();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(".preview_section_right_1").hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('.preview_section_right_2').show();
                $('#result').fadeIn(600);
                $('#result').text(data);
                result_data = data;
                console.log('Success!:' + data);
                MathJax.typeset();
            },
        });
    });

    $('#copy_button').click(function () {
        $('#copy_button').tooltip();
        $('#copy_button').tooltip('show');
        navigator.clipboard.writeText(result_data);
        setTimeout(function() {
            $('#copy_button').tooltip('hide');
          }, 1000);
    });

    $('#redo_button').click(function () {
        $('.preview_section_right_2').hide();
        $('.preview_section_right_1').show();
        $('.upload_section').show();
        $('.preview_section').hide();
    });

});
