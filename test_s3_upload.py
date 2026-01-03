
import auth


token = auth.get_sh_token()  # Ensure token fetching works at startup
print("server token prefix:", token)

